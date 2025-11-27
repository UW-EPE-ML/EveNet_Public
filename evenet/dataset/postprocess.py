import numpy as np
import torch
from decimal import Decimal, getcontext
from collections import OrderedDict
import warnings

from torch import Tensor


def masked_stats(arr, weights=None):
    mask = arr != 0

    if weights is None:
        factor = mask  # 1 for valid entries, 0 for masked out
    else:
        weights = np.asarray(weights)
        if weights.ndim == 1:
            weights = weights[:, None]  # broadcast over columns
        factor = weights * mask  # zero where masked out

    values = arr * mask
    w_values = values * factor  # = arr * mask * factor = arr * factor

    sum_ = w_values.sum(axis=0)
    sumsq = (w_values ** 2).sum(axis=0)
    count = factor.sum(axis=0)

    return {"sum": sum_, "sumsq": sumsq, "count": count}


def compute_effective_counts_from_freq_decimal(freqs: list[float], precision: int = 50) -> np.ndarray:
    """
    Compute class-balanced weights based on effective number of samples using decimal for stability.

    Args:
        freqs (list[float]): List of sample counts per class.
        precision (int): Number of decimal places to use.

    Returns:
        list[float]: Class weights normalized to sum ≈ number of classes.
    """
    getcontext().prec = precision
    freqs = [Decimal(f) for f in freqs]
    N = sum(freqs)

    if N == 0:
        raise ValueError("Total number of samples is zero. Check input frequencies.")

    beta = Decimal(1) - Decimal(1) / N

    effective_num = []
    for f in freqs:
        if f == 0:
            effective_num.append(Decimal('Infinity'))  # or float('inf')
        else:
            numerator = Decimal(1) - beta ** f
            denominator = Decimal(1) - beta
            effective = numerator / denominator
            effective_num.append(effective)

    weights = [Decimal(1) / e if e != 0 else Decimal(0) for e in effective_num]
    total = sum(weights)
    normalized = [(w * len(freqs)) / total for w in weights]

    return np.array([float(w) for w in normalized], dtype=np.float32)


def compute_classification_balance(class_counts: np.ndarray) -> np.ndarray:
    """
    Wrapper to compute effective class weights from raw class frequency counts.
    """
    class_counts = np.asarray(class_counts, dtype=np.float64)

    if np.all(class_counts < 1):
        warnings.warn(
            "All class counts are < 1. Assuming these are fractions. "
            "Reweighting by simple inverse ratio to the largest."
        )
        max_val = np.max(class_counts)
        if max_val == 0:
            raise ValueError("All class counts are zero. Cannot compute balance.")
        weights = max_val / class_counts
        # normalize to sum ≈ num_classes for consistency
        weights *= len(class_counts) / np.sum(weights)
        return weights.astype(np.float32)

    # Some counts < 1 but not all: clip them to 1
    if np.any(class_counts < 1):
        warnings.warn(
            "Some class counts are < 1. Clipping them to 1 to avoid instability in effective number calculation."
        )
        class_counts = np.where(class_counts < 1, 1.0, class_counts)

    return compute_effective_counts_from_freq_decimal(class_counts.tolist(), precision=50)


def merge_stat_maps(stats_list, key):
    """Merge stats only if key exists inside stats_list."""
    present = [s[key] for s in stats_list if key in s]
    if not present:
        return None

    # merge sums
    merged = {k: sum(d[k] for d in present) for k in ["sum", "sumsq", "count"]}

    # compute final mean/std
    count = merged["count"]
    safe_count = np.where(count == 0, 1, count)
    mean = merged["sum"] / safe_count
    variance = merged["sumsq"] / safe_count - mean ** 2
    variance = np.clip(variance, 0, None)
    std = np.sqrt(variance)

    mean = np.where(count == 0, 0, mean)
    std = np.where(std == 0, 1, std)

    return {"mean": mean, "std": std, "count": count}


def merge_simple_counts(stats_list, key):
    present = [s[key] for s in stats_list if key in s]
    if not present:
        return None
    return np.sum(present, axis=0)


def merge_assignment_masks(list_of_assignment_masks):
    # assignment_mask: dict[str -> list[dict[str -> np.ndarray]]]
    merged = {}

    all_processes = list_of_assignment_masks[0].keys()

    for process in all_processes:
        collected_dicts = []
        for am in list_of_assignment_masks:
            if process in am:
                collected_dicts.extend(am[process])

        # merge list of dicts → dict of concatenated arrays
        if not collected_dicts:
            merged[process] = {}
            continue

        keys = collected_dicts[0].keys()
        temp = {k: [] for k in keys}
        for d in collected_dicts:
            for k in keys:
                temp[k].append(d[k])
        merged[process] = {k: np.concatenate(v, axis=0) for k, v in temp.items()}

    return merged


def compute_particle_balance(merged_assignment_masks, event_equivalence_classes, precision: int = 50):
    getcontext().prec = precision

    particle_balance = OrderedDict()
    common_processes = [
        key for key in event_equivalence_classes.keys()
        if key in merged_assignment_masks
    ]

    for process in common_processes:
        assignment_masks = merged_assignment_masks[process]

        if not assignment_masks:
            print(f"[compute_particle_balance] Skipping process '{process}' (empty assignment masks)")
            continue

        # Check that all values are torch/numpy arrays with matching length
        keys = list(assignment_masks.keys())
        lengths = [len(v) for v in assignment_masks.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"[{process}] Inconsistent mask lengths: {dict(zip(keys, lengths))}")

        # Build tensor: shape (num_targets, num_events)
        try:
            masks = torch.stack([torch.tensor(assignment_masks[k], dtype=torch.bool) for k in keys])
        except Exception as e:
            raise RuntimeError(f"[{process}] Failed to stack assignment masks: {e}")

        num_targets = len(keys)
        num_events = masks.shape[1]
        full_targets = frozenset(range(num_targets))

        # Start computing equivalence class weights
        eq_class_counts = {}

        for eq_class in event_equivalence_classes[process]:
            eq_class_count = 0
            for positive_target in eq_class:
                negative_target = full_targets - positive_target

                positive_target = masks[list(positive_target), :].all(0)
                negative_target = masks[list(negative_target), :].any(0)
                targets = positive_target & ~negative_target

                eq_class_count += targets.sum().item()
            eq_class_counts[eq_class] = eq_class_count + 1

        # Compute class-balanced weights
        # Ensure num_events is passed in as int or string (not float!)
        num_events = Decimal(str(num_events))
        beta = Decimal('1') - (Decimal('10') ** (-num_events.log10()))
        eq_class_weights = {
            key: (Decimal('1') - beta) / (Decimal('1') - beta ** Decimal(value))
            for key, value in eq_class_counts.items()
        }
        target_weights = {
            target: float(weight)  # Convert back to float if needed later
            for eq_class, weight in eq_class_weights.items()
            for target in eq_class
        }
        index_tensor = 2 ** torch.arange(num_targets)
        target_weights_tensor = torch.zeros(2 ** num_targets)

        norm = float(sum(eq_class_weights.values()))
        for target, weight in target_weights.items():
            index = index_tensor[list(target)].sum()
            target_weights_tensor[index] = len(eq_class_weights) * weight / norm

        particle_balance[process] = (index_tensor, target_weights_tensor)

    return particle_balance


class PostProcessor:
    def __init__(self, global_config):
        self.stats = []
        self.assignment_mask = {p: [] for p in global_config.event_info.process_names}
        self.event_equivalence_classes = global_config.event_info.event_equivalence_classes

        # Feature flags (auto-detected from first .add())
        self.use_regression = False
        self.use_classification = False
        self.use_subprocess = False
        self.use_assignment = False
        self.use_segmentation = False
        self.use_invisible = False

        self._initialized = False

    def add(
            self,
            x,
            conditions,
            num_vectors,
            regression=None,
            class_counts=None,
            subprocess_counts=None,
            invisible=None,
            segment_class_counts=None,
            segment_full_class_counts=None,
            segment_regression=None,
    ):

        # ---- Required keys ----
        record = {
            "x": masked_stats(x.reshape(-1, x.shape[-1])),
            "conditions": masked_stats(conditions),
            "input_num": masked_stats(num_vectors.reshape(-1, num_vectors.shape[-1])),
        }

        # ---- Auto-detect optional components ----
        if not self._initialized:
            self.use_regression = regression is not None
            self.use_classification = class_counts is not None
            self.use_assignment = subprocess_counts is not None
            self.use_segmentation = segment_class_counts is not None
            self.use_invisible = invisible is not None
            self._initialized = True

        # ---- Optional: regression ----
        if self.use_regression and regression is not None:
            record["regression"] = masked_stats(regression)

        # ---- Optional: classification ----
        if self.use_classification and class_counts is not None:
            record["class_counts"] = class_counts

        # ---- Optional: assignment ----
        if self.use_assignment and subprocess_counts is not None:
            record["subprocess_counts"] = subprocess_counts

        # ---- Optional: invisible ----
        if self.use_invisible and invisible is not None:
            inv = invisible.reshape(-1, invisible.shape[-1])
            record["invisible"] = masked_stats(inv)

        # ---- Optional: segmentation ----
        if self.use_segmentation and segment_class_counts is not None:
            record["segment_class_counts"] = segment_class_counts
            record["segment_full_class_counts"] = segment_full_class_counts
            record["segment_regression"] = masked_stats(segment_regression)

        self.stats.append(record)

    def add_assignment_mask(self, process, dict_particle):
        self.assignment_mask[process].append(dict_particle)

    @classmethod
    def merge(cls, instances, regression_names=None, saved_results_path=None):

        instances = [i for i in instances if i is not None]
        stats_list = [s for inst in instances for s in inst.stats]
        first = instances[0]

        merged = {}

        # ---- Required stats ----
        merged["x"] = merge_stat_maps(stats_list, "x")
        merged["conditions"] = merge_stat_maps(stats_list, "conditions")
        merged["input_num"] = merge_stat_maps(stats_list, "input_num")

        # ---- Optional stats ----
        if first.use_regression:
            merged["regression"] = merge_stat_maps(stats_list, "regression")

        if first.use_classification:
            merged["class_counts"] = merge_simple_counts(stats_list, "class_counts")
            merged["class_balance"] = compute_classification_balance(merged["class_counts"])

        if first.use_invisible:
            merged["invisible"] = merge_stat_maps(stats_list, "invisible")

        if first.use_segmentation:
            merged["segment_class_counts"] = merge_simple_counts(stats_list, "segment_class_counts")
            merged["segment_full_class_counts"] = merge_simple_counts(stats_list, "segment_full_class_counts")
            merged["segment_regression"] = merge_stat_maps(stats_list, "segment_regression")

            merged["segment_class_balance"] = compute_classification_balance(
                merged["segment_class_counts"]
            )
            merged["segment_full_class_balance"] = compute_classification_balance(
                merged["segment_full_class_counts"]
            )

        if first.use_assignment:
            merged["subprocess_counts"] = merge_simple_counts(stats_list, "subprocess_counts")
            merged["subprocess_balance"] = compute_classification_balance(merged["subprocess_counts"])

            all_masks = [inst.assignment_mask for inst in instances]
            # You keep your own merge + particle_balance implementation
            merged_masks = merge_assignment_masks(all_masks)
            particle_balance = compute_particle_balance(
                merged_masks,
                first.event_equivalence_classes
            )
            merged["particle_balance"] = particle_balance

        ########################################################
        # Final output – only include keys that exist
        ########################################################
        output: dict[str, Tensor | dict[str, Tensor]] = {
            "input_mean": {
                "Source": torch.tensor(merged["x"]["mean"], dtype=torch.float32),
                "Conditions": torch.tensor(merged["conditions"]["mean"], dtype=torch.float32)
            },
            "input_std": {
                "Source": torch.tensor(merged["x"]["std"], dtype=torch.float32),
                "Conditions": torch.tensor(merged["conditions"]["std"], dtype=torch.float32)
            },
            "input_num_mean": {"Source": torch.tensor(merged["input_num"]["mean"], dtype=torch.float32)},
            "input_num_std": {"Source": torch.tensor(merged["input_num"]["std"], dtype=torch.float32)},
        }

        if first.use_regression:
            output["regression_mean"] = {
                name: torch.tensor(merged["regression"]["mean"][i]) for i, name in enumerate(regression_names)
            }
            output["regression_std"] = {
                name: torch.tensor(merged["regression"]["std"][i]) for i, name in enumerate(regression_names)
            }

        if first.use_classification:
            output["class_counts"] = torch.tensor(merged["class_counts"])
            output["class_balance"] = torch.tensor(merged["class_balance"])

        if first.use_assignment:
            output["subprocess_counts"] = torch.tensor(merged["subprocess_counts"])
            output["subprocess_balance"] = torch.tensor(merged["subprocess_balance"])
            output["particle_balance"] = merged["particle_balance"]

        if first.use_segmentation:
            output["segment_class_counts"] = torch.tensor(merged["segment_class_counts"])
            output["segment_class_balance"] = torch.tensor(merged["segment_class_balance"])
            output["segment_full_class_counts"] = torch.tensor(merged["segment_full_class_counts"])
            output["segment_full_class_balance"] = torch.tensor(merged["segment_full_class_balance"])
            output["segment_regression_mean"] = torch.tensor(merged["segment_regression"]["mean"])
            output["segment_regression_std"] = torch.tensor(merged["segment_regression"]["std"])

        if first.use_invisible:
            output["invisible_mean"] = {"Source": torch.tensor(merged["invisible"]["mean"])}
            output["invisible_std"] = {"Source": torch.tensor(merged["invisible"]["std"])}

        # Save
        if saved_results_path:
            torch.save(output, f"{saved_results_path}/normalization.pt")

        return output
