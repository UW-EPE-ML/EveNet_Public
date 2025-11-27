import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evenet.dataset.preprocess import flatten_dict
from evenet.dataset.postprocess import PostProcessor
from preprocessing.sanity_checks import InputDictionarySanityChecker

# ======================================================================
# Logging
# ======================================================================

logger = logging.getLogger(__name__)


# ======================================================================
# Utility Helpers
# ======================================================================

def load_npz(path):
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def event_split_indices(n_events, ratio, rng=None):
    """
    Given number of events, return indices for train, val, test splits.
    ratio = (train_ratio, val_ratio, test_ratio)
    """
    if not np.isclose(sum(ratio), 1.0):
        raise ValueError(f"Split ratio must sum to 1, got {ratio}")

    rng = rng or np.random.default_rng(42)
    indices = rng.permutation(n_events)

    n_train = int(ratio[0] * n_events)
    n_val = int(ratio[1] * n_events)
    # test = remainder

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def slice_event_dict(data, idx, n_events):
    """
    Slice a dictionary containing event-wise arrays.
    If an array has length == n_events, slice along axis 0.
    Otherwise keep it unchanged (metadata).
    """
    out = {}
    for key, arr in data.items():
        if hasattr(arr, "shape") and arr.shape[0] == n_events:
            out[key] = arr[idx]
        else:
            out[key] = arr
    return out


# ======================================================================
# Sanity Checks
# ======================================================================

sanity_checker = InputDictionarySanityChecker()


# ======================================================================
# Processing Logic
# ======================================================================

def process_dict(
        pdict,
        *,
        global_config,
        unique_process_ids,
        assignment_keys,
        statistics,
        shape_metadata,
        store_chunks,
):
    """
    Process a single *per-event dictionary* (after splitting or direct loading).
    Update shape metadata, statistics, and append arrow chunk.
    """

    if all(len(arr) == 0 for arr in pdict.values()):
        return shape_metadata

    sanity_checker.run(pdict, global_config=global_config)

    flattened, meta = flatten_dict(pdict)

    # --- Metadata consistency ---
    if shape_metadata is None:
        shape_metadata = meta
    else:
        assert shape_metadata == meta, "Shape metadata mismatch."

    # store table chunk
    store_chunks.append(flattened)

    # --- If no statistics needed (val/test), we are done ---
    if statistics is None:
        return shape_metadata

    # --- Compute statistics (train only) ---
    # ================================================================
    # EVENT WEIGHTS
    # ================================================================
    if "event_weight" in pdict:
        weights = pdict["event_weight"]
        weights = weights.astype(np.float32)
    else:
        # No weight available → assume all weights = 1
        # Determine number of events from ANY event-level array
        # (the ones with first dimension matching events)
        n_events = None
        for arr in pdict.values():
            if hasattr(arr, "shape") and len(arr.shape) > 0:
                if n_events is None:
                    n_events = arr.shape[0]
                else:
                    n_events = min(n_events, arr.shape[0])
        if n_events is None:
            raise RuntimeError("Cannot infer number of events without event_weight or event-level arrays.")
        weights = np.ones(n_events, dtype=np.float32)

    # Reshape for later multiplication
    w = weights.reshape(-1, 1)

    # ==> CLASSIFICATION
    if len(unique_process_ids) > 0:
        # TODO: Fix me
        proc_info = global_config.process_info[process]

        class_counts = np.zeros(len(unique_process_ids), dtype=np.float32)
        class_counts[proc_info["process_id"]] = np.sum(weights)

    else:
        class_counts = None  # or np.array([])

    # ==> SEGMENTATION
    seg_class_cnt = None
    seg_full_cnt = None

    if "segmentation-class" in pdict:
        seg_class_cnt = np.sum(
            np.sum(pdict["segmentation-class"], axis=1) * w,
            axis=0
        )

    if "segmentation-full-class" in pdict:
        seg_full_cnt = np.sum(
            np.sum(pdict["segmentation-full-class"], axis=1) * w,
            axis=0
        )

    # ==> ASSIGNMENTS
    process_names = global_config.event_info.process_names
    subprocess_counts = None

    if "assignments-mask" in pdict and len(assignment_keys) > 0:
        subprocess_counts = np.zeros(len(process_names), dtype=np.float32)

        evt_proc = pdict["process_names"]
        N = len(evt_proc)

        # loop over each process that appears in the events
        for i, proc_name in enumerate(process_names):
            mask = {}

            # select events belonging to this process
            valid = (evt_proc == proc_name)
            if not np.any(valid):
                continue

            subprocess_counts[i] = np.sum(weights[valid])

            # all assignment keys for THIS process, in the order matching columns 0..3
            proc_keys = [
                k for k in assignment_keys
                if f"TARGETS/{proc_name}/" in k or f"LABELS/{proc_name}/" in k
            ]

            # now local_idx is the column index into assignments_mask
            for local_idx, key in enumerate(proc_keys):
                particle = key.split("/")[2]  # "t1", "t2", ...

                # lazily allocate full-length vector per particle
                if particle not in mask:
                    mask[particle] = np.zeros(N, dtype=np.float32)

                mask[particle][valid] = (
                        pdict["assignments-mask"][valid, local_idx] * weights[valid]
                )

            statistics.add_assignment_mask(proc_name, mask)

    statistics.add(
        x=pdict['x'],
        conditions=pdict['conditions'],
        num_vectors=pdict['num_sequential_vectors'],
        regression=pdict['regression-data'] if "regression-data" in pdict else None,
        class_counts=class_counts,
        subprocess_counts=subprocess_counts,
        invisible=pdict['x_invisible'] if "x_invisible" in pdict else None,
        segment_class_counts=seg_class_cnt,
        segment_full_class_counts=seg_full_cnt,
        segment_regression=pdict["segmentation-momentum"] if "segmentation-momentum" in pdict else None,
    )

    return shape_metadata


# ======================================================================
# Main Preprocessing (supports file-level splitting + event-level splitting)
# ======================================================================

def preprocess(
        *,
        files=None,
        train=None,
        val=None,
        test=None,
        split_ratio=(1.0, 0.0, 0.0),
        store_dir,
        global_config,
        unique_process_ids,
        assignment_keys,
        verbose=True,
):
    """
    Unified preprocessing.

    Supported usage:
        preprocess(files=[...], split_ratio=(0.8,0.1,0.1))  # event-level split
        preprocess(train=[...], val=[...], test=[...])      # explicit file-level split
        preprocess(files="onefile.npz")
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # shape metadata shared across all splits
    shape_metadata = None

    # Arrow table chunks for each split
    tables = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Train statistics only
    train_stats = PostProcessor(global_config)

    # ==================================================================
    # Case 1: explicit file-level splits
    # ==================================================================
    if any([train, val, test]):
        # ---------------- TRAIN ----------------
        for f in ensure_list(train):
            if verbose: print(f"[TRAIN] Loading {f}")
            data = load_npz(f)
            shape_metadata = process_dict(
                data,
                global_config=global_config,
                unique_process_ids=unique_process_ids,
                assignment_keys=assignment_keys,
                statistics=train_stats,
                shape_metadata=shape_metadata,
                store_chunks=tables["train"],
            )

        # ---------------- VAL ----------------
        for f in ensure_list(val):
            if verbose: print(f"[VAL] Loading {f}")
            data = load_npz(f)
            shape_metadata = process_dict(
                data,
                global_config=global_config,
                unique_process_ids=unique_process_ids,
                assignment_keys=assignment_keys,
                statistics=None,
                shape_metadata=shape_metadata,
                store_chunks=tables["val"],
            )

        # ---------------- TEST ----------------
        for f in ensure_list(test):
            if verbose: print(f"[TEST] Loading {f}")
            data = load_npz(f)
            shape_metadata = process_dict(
                data,
                global_config=global_config,
                unique_process_ids=unique_process_ids,
                assignment_keys=assignment_keys,
                statistics=None,
                shape_metadata=shape_metadata,
                store_chunks=tables["test"],
            )

    # ==================================================================
    # Case 2: event-level split (files + split_ratio)
    # ==================================================================
    else:
        rng = np.random.default_rng(42)
        for f in ensure_list(files):
            if verbose: print(f"[SPLIT] Loading {f}")
            data = load_npz(f)

            n_events = len(data["x"])
            train_idx, val_idx, test_idx = event_split_indices(n_events, split_ratio, rng)

            split_data = {
                "train": slice_event_dict(data, train_idx, n_events),
                "val": slice_event_dict(data, val_idx, n_events),
                "test": slice_event_dict(data, test_idx, n_events),
            }

            for split_name, pdict in split_data.items():
                if pdict is None:
                    continue

                shape_metadata = process_dict(
                    pdict,
                    global_config=global_config,
                    unique_process_ids=unique_process_ids,
                    assignment_keys=assignment_keys,
                    statistics=train_stats if split_name == "train" else None,
                    shape_metadata=shape_metadata,
                    store_chunks=tables[split_name],
                )

    # ==================================================================
    # Save all outputs
    # ==================================================================

    for split_name, chunks in tables.items():
        if len(chunks) == 0:
            continue

        table = pa.concat_tables(chunks)
        shuffle_idx = np.random.default_rng(31).permutation(table.num_rows)
        table = table.take(pa.array(shuffle_idx))

        out_path = store_dir / f"{split_name}.parquet"
        pq.write_table(table, out_path)
        if verbose:
            print(f"[INFO] Saved {split_name} → {out_path} ({table.nbytes / 1024 / 1024:.2f} MB)")

    # save shared metadata
    with open(store_dir / "shape_metadata.json", "w") as f:
        json.dump(shape_metadata, f)

    PostProcessor.merge([train_stats], saved_results_path=store_dir)
