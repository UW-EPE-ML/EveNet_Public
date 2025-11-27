import numpy as np
import torch
from decimal import Decimal, getcontext
from collections import OrderedDict
import warnings


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


def compute_effective_counts_from_freq(freqs: np.ndarray) -> np.ndarray:
    """
    Compute class-balanced weights based on effective number of samples.
    Ref: https://arxiv.org/pdf/1901.05555.pdf

    Args:
        freqs (np.ndarray): Array of sample counts per class. Index is class label.

    Returns:
        np.ndarray: Class weights normalized so that sum ≈ number of classes.
    """
    # TODO: check numerical stability

    freqs = freqs.astype(np.longdouble)
    N = freqs.sum()
    if N == 0:
        raise ValueError("Total number of samples is zero. Check input frequencies.")

    beta = 1 - (1 / N)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Avoid direct power to prevent underflow
        log_beta = np.log(beta)
        power_term = np.exp(freqs * log_beta)
        effective_num = (1.0 - power_term) / (1.0 - beta)

        weights = 1.0 / effective_num
        weights[~np.isfinite(weights)] = 0.0  # fix nan/inf
        weights = weights * len(freqs) / weights.sum()  # Normalize to total class count

    return weights


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


def merge_stats(stats_list):
    def merge_two(a, b):
        return {
            "sum": a["sum"] + b["sum"],
            "sumsq": a["sumsq"] + b["sumsq"],
            "count": a["count"] + b["count"]
        }

    def compute_mean_std(agg):
        count = agg["count"]
        sum_ = agg["sum"]
        sumsq = agg["sumsq"]

        # Avoid divide-by-zero
        safe_count = np.where(count == 0, 1, count)

        mean = sum_ / safe_count
        variance = sumsq / safe_count - mean ** 2
        variance = np.clip(variance, a_min=0.0, a_max=None)
        std = np.sqrt(variance)

        # Set mean = 0, std = 1 for features with no data
        mean = np.where(count == 0, 0.0, mean)
        std = np.where(count == 0, 1.0, std)
        std = np.where(std == 0, 1.0, std)

        return {'mean': mean, 'std': std}

    # Accumulate across all files
    total = {
        "x": None,
        "conditions": None,
        "regression": None,
        "input_num": None,
        "invisible": None,
        'segment_regression': None,
    }

    for s in stats_list:
        for key in total.keys():
            if total[key] is None:
                total[key] = s[key]
            else:
                total[key] = merge_two(total[key], s[key])

    total['class_counts'] = np.sum([s["class_counts"] for s in stats_list], axis=0)
    total['subprocess_counts'] = np.sum([s["subprocess_counts"] for s in stats_list], axis=0)
    total['segment_class_counts'] = np.sum([s["segment_class_counts"] for s in stats_list], axis=0)
    total['segment_full_class_counts'] = np.sum([s["segment_full_class_counts"] for s in stats_list], axis=0)

    # Final result
    result = {
        "x": compute_mean_std(total["x"]),
        "conditions": compute_mean_std(total["conditions"]),
        "regression": compute_mean_std(total["regression"]),
        "input_num": compute_mean_std(total["input_num"]),
        "class_counts": total["class_counts"],
        "class_balance": compute_classification_balance(total["class_counts"]),
        "subprocess_counts": total["subprocess_counts"],
        "subprocess_balance": compute_classification_balance(total["subprocess_counts"]),

        "segment_class_counts": total["segment_class_counts"],
        "segment_class_balance": compute_classification_balance(total["segment_class_counts"]),
        "segment_full_class_counts": total["segment_full_class_counts"],
        "segment_full_class_balance": compute_classification_balance(total["segment_full_class_counts"]),
        "segment_regression": compute_mean_std(total["segment_regression"]),

        "invisible": compute_mean_std(total["invisible"]),
    }
    return result


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

    def add(
            self, x, conditions, regression, num_vectors, class_counts, subprocess_counts, invisible,
            segment_class_counts,segment_full_class_counts, segment_regression,
            event_weight=None
    ):
        x_stats = masked_stats(x.reshape(-1, x.shape[-1]), None)
        cond_stats = masked_stats(conditions, None)
        regression_stats = masked_stats(regression, None)
        num_vectors_stats = masked_stats(num_vectors, None)
        segment_regression_stats = masked_stats(segment_regression, None)

        if invisible.size == 0:
            reshaped = np.empty((0, invisible.shape[-1]))  # safe manual reshape
        else:
            reshaped = invisible.reshape(-1, invisible.shape[-1])
        invisible_stats = masked_stats(reshaped,  None)

        self.stats.append({
            "x": x_stats,
            "conditions": cond_stats,
            "regression": regression_stats,
            "input_num": num_vectors_stats,
            "class_counts": class_counts,
            "subprocess_counts": subprocess_counts,
            "segment_class_counts": segment_class_counts,
            "segment_full_class_counts": segment_full_class_counts,
            "segment_regression": segment_regression_stats,

            "invisible": invisible_stats,
        })

    def add_assignment_mask(self, process, dict_particle):
        self.assignment_mask[process].append(dict_particle)

    @classmethod
    def merge(
            cls,
            instances,
            regression_names,
            saved_results_path=None,
    ):
        # Filter out None instances, when a run dir does not contain any data
        # for the desired physics processes
        valid_instances = [inst for inst in instances if inst is not None]
        combined = [item for a in valid_instances for item in a.stats]
        merged_stats = merge_stats(combined)

        all_assignment_masks = [inst.assignment_mask for inst in valid_instances]
        merged_assignment_masks = merge_assignment_masks(all_assignment_masks)
        particle_balance = compute_particle_balance(merged_assignment_masks, instances[0].event_equivalence_classes)

        saved_results = {
            'input_mean': {
                'Source': torch.tensor(merged_stats["x"]["mean"], dtype=torch.float32),
                'Conditions': torch.tensor(merged_stats["conditions"]["mean"], dtype=torch.float32),
            },
            'input_std': {
                'Source': torch.tensor(merged_stats["x"]["std"], dtype=torch.float32),
                'Conditions': torch.tensor(merged_stats["conditions"]["std"], dtype=torch.float32),
            },
            'input_num_mean': {
                'Source': torch.tensor(merged_stats["input_num"]["mean"], dtype=torch.float32)
            },
            'input_num_std': {
                'Source': torch.tensor(merged_stats["input_num"]["std"], dtype=torch.float32)
            },
            'regression_mean': {
                k: torch.tensor(merged_stats["regression"]["mean"][i], dtype=torch.float32)
                for i, k in enumerate(regression_names)
            },
            'regression_std': {
                k: torch.tensor(merged_stats["regression"]["std"][i], dtype=torch.float32)
                for i, k in enumerate(regression_names)
            },
            'class_counts': torch.tensor(merged_stats["class_counts"], dtype=torch.float32),
            'class_balance': torch.tensor(merged_stats["class_balance"], dtype=torch.float32),
            'particle_balance': particle_balance,
            'subprocess_counts': torch.tensor(merged_stats["subprocess_counts"], dtype=torch.float32),
            'subprocess_balance': torch.tensor(merged_stats["subprocess_balance"], dtype=torch.float32),

            'segment_class_counts': torch.tensor(merged_stats["segment_class_counts"], dtype=torch.float32),
            'segment_class_balance': torch.tensor(merged_stats["segment_class_balance"], dtype=torch.float32),

            'segment_full_class_counts': torch.tensor(merged_stats["segment_full_class_counts"], dtype=torch.float32),
            'segment_full_class_balance': torch.tensor(merged_stats["segment_full_class_balance"], dtype=torch.float32),

            'segment_regression_mean': torch.tensor(merged_stats["segment_regression"]["mean"], dtype=torch.float32),
            'segment_regression_std': torch.tensor(merged_stats["segment_regression"]["std"], dtype=torch.float32),

            'invisible_mean': {
                'Source': torch.tensor(merged_stats["invisible"]["mean"], dtype=torch.float32),
            },
            'invisible_std': {
                'Source': torch.tensor(merged_stats["invisible"]["std"], dtype=torch.float32),
            },
        }

        if saved_results_path:
            torch.save(saved_results, f"{saved_results_path}/normalization.pt")
