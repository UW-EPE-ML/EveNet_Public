import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evenet.dataset.preprocess import flatten_dict
from evenet.dataset.postprocess import PostProcessor
from evenet.dataset.types import InputType
from preprocessing.sanity_checks import InputDictionarySanityChecker

# ======================================================================
# Logging
# ======================================================================

logger = logging.getLogger(__name__)


# ======================================================================
# Utility Helpers
# ======================================================================

@dataclass
class LogScalePlan:
    sequential_indices: list[int] = field(default_factory=list)
    sequential_names: list[str] = field(default_factory=list)
    condition_indices: list[int] = field(default_factory=list)
    condition_names: list[str] = field(default_factory=list)
    invisible_indices: list[int] = field(default_factory=list)
    invisible_names: list[str] = field(default_factory=list)

    def description(self) -> str:
        rows = [
            (
                "x",
                ", ".join(self.sequential_names) if self.sequential_names else "<none>",
            ),
            (
                "conditions",
                ", ".join(self.condition_names) if self.condition_names else "<none>",
            ),
            (
                "x_invisible",
                ", ".join(self.invisible_names) if self.invisible_names else "<none>",
            ),
        ]

        header = ("Input", "Features (log1p)")
        column_widths = [
            max(len(header[0]), *(len(row[0]) for row in rows)),
            max(len(header[1]), *(len(row[1]) for row in rows)),
        ]

        def _format_row(left: str, right: str) -> str:
            return f"{left.ljust(column_widths[0])} | {right}"

        formatted = [
            "\n",
            _format_row(*header),
            "-" * column_widths[0] + "-+-" + "-" * column_widths[1],
            *(_format_row(*row) for row in rows),
        ]

        return "\n".join(formatted)


def load_npz(path):
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def build_log_scale_plan(global_config) -> LogScalePlan:
    plan = LogScalePlan()
    event_info = global_config.event_info

    seq_offset = 0
    cond_offset = 0

    for input_name, features in event_info.input_features.items():
        input_type = str(event_info.input_types[input_name]).upper()

        if input_type == InputType.Sequential.value:
            for idx, feature in enumerate(features):
                if feature.log_scale:
                    plan.sequential_indices.append(seq_offset + idx)
                    plan.sequential_names.append(f"{input_name}:{feature.name}")
            seq_offset += len(features)

        elif input_type == InputType.Global.value:
            for idx, feature in enumerate(features):
                if feature.log_scale:
                    plan.condition_indices.append(cond_offset + idx)
                    plan.condition_names.append(f"{input_name}:{feature.name}")
            cond_offset += len(features)

    for idx, feature in enumerate(event_info.invisible_input_features):
        if feature.log_scale:
            plan.invisible_indices.append(idx)
            plan.invisible_names.append(f"Invisible:{feature.name}")

    return plan


def _validate_log_values(values: np.ndarray, column_name: str):
    if not np.isfinite(values).all():
        invalid_count = np.count_nonzero(~np.isfinite(values))
        raise ValueError(
            f"Found {invalid_count} non-finite values in {column_name} before log scaling"
        )

    if (values < 0).any():
        min_value = values.min()
        raise ValueError(
            f"Negative values detected in {column_name} before log scaling (min={min_value})"
        )


def apply_log_scaling(pdict: dict, plan: LogScalePlan) -> dict:
    def _apply(arr: np.ndarray, indices: list[int], names: list[str], key: str) -> np.ndarray:
        for idx, name in zip(indices, names):
            values = arr[..., idx]
            _validate_log_values(values, f"{key}:{name}")
            arr[..., idx] = np.log1p(values).astype(arr.dtype, copy=False)
        return arr

    if plan.sequential_indices and "x" in pdict:
        pdict["x"] = _apply(pdict["x"], plan.sequential_indices, plan.sequential_names, "x")

    if plan.condition_indices and "conditions" in pdict:
        pdict["conditions"] = _apply(
            pdict["conditions"], plan.condition_indices, plan.condition_names, "conditions"
        )

    if plan.invisible_indices and "x_invisible" in pdict:
        pdict["x_invisible"] = _apply(
            pdict["x_invisible"], plan.invisible_indices, plan.invisible_names, "x_invisible"
        )

    return pdict


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
        log_scale_plan,
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

    pdict = apply_log_scaling(pdict, log_scale_plan)

    sanity_checker.run(pdict, global_config=global_config)

    # --- If no statistics needed (val/test), we are done ---
    if statistics is not None:

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
            class_counts = np.bincount(pdict['classification'], weights=weights)
            unweighted_class_counts = np.bincount(pdict['classification'])

            lines = [
                "===============================================",
                " idx | Class          | Weighted Sum | Events ",
                "==============================================="
            ]
            for idx, name in enumerate(unique_process_ids):
                lines.append(f"{idx:3d} | {name:14s} | {class_counts[idx]:12.3f} | {unweighted_class_counts[idx]:6d}")
            lines.append("===============================================")
            table_text = "\n".join(lines)
            # PRINT ELEGANTLY
            logger.info("\n%s", table_text)
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

        pdict.pop("process_names", None)

    flattened, meta = flatten_dict(pdict)
    # store table chunk
    store_chunks.append(flattened)
    if statistics is None:
        return shape_metadata

    # --- Metadata consistency ---
    if shape_metadata is None:
        shape_metadata = meta
    else:
        assert shape_metadata == meta, "Shape metadata mismatch."

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

    log_scale_plan = build_log_scale_plan(global_config)
    logger.info("Applying np.log1p to log-scale features: %s", log_scale_plan.description())

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
                log_scale_plan=log_scale_plan,
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
                log_scale_plan=log_scale_plan,
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
                log_scale_plan=log_scale_plan,
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
                    log_scale_plan=log_scale_plan,
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
