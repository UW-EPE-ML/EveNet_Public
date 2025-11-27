import logging
from dataclasses import dataclass
from typing import Iterable


logger = logging.getLogger(__name__)


@dataclass
class ExpectedTensor:
    name: str
    expected_shape: tuple
    description: str


def _format_shape(shape: Iterable[int | None] | str) -> str:
    if isinstance(shape, str):
        return shape
    return "(" + ", ".join("?" if d is None else str(d) for d in shape) + ")"


def _render_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _row(cols: list[str]) -> str:
        return "|" + "|".join(f" {c.ljust(w)} " for c, w in zip(cols, widths)) + "|"

    rendered = [title, sep, _row(headers), sep]
    rendered.extend(_row(r) for r in rows)
    rendered.append(sep)
    return "\n".join(rendered)


class InputDictionarySanityChecker:
    """Validate and describe incoming preprocessing dictionaries with rich logging."""

    INPUT_TENSORS: tuple[ExpectedTensor, ...] = (
        ExpectedTensor("num_vectors", (None,), "Total global + sequential objects per event"),
        ExpectedTensor("num_sequential_vectors", (None,), "Sequential entries per event"),
        ExpectedTensor("x", (None, 18, 7), "Point-cloud tensor (18 particles × 7 features)"),
        ExpectedTensor("x_mask", (None, 18), "Mask for point-cloud slots"),
        ExpectedTensor("conditions", (None, None), "Event-level scalars"),
        ExpectedTensor("conditions_mask", (None, 1), "Mask for event-level scalars"),
    )

    TASK_TENSORS: dict[str, tuple[ExpectedTensor, ...]] = {
        "classification": (
            ExpectedTensor("classification", (None,), "Per-event class label"),
            ExpectedTensor("event_weight", (None,), "Optional event weight"),
        ),
        "truth_generation": (
            ExpectedTensor("x_invisible", (None, None, None), "Invisible particle features"),
            ExpectedTensor("x_invisible_mask", (None, None), "Mask for invisible particles"),
            ExpectedTensor("num_invisible_raw", (None,), "Raw count of invisibles"),
            ExpectedTensor("num_invisible_valid", (None,), "Valid invisibles after matching"),
        ),
        "resonance_assignment": (
            ExpectedTensor("assignments-indices", (None, None, None), "Resonance-to-child mapping"),
            ExpectedTensor("assignments-mask", (None, None), "Resonance validity mask"),
            ExpectedTensor("assignments-indices-mask", (None, None, None), "Per-child mask"),
        ),
        "segmentation": (
            ExpectedTensor("segmentation-class", (None, None, None), "One-hot daughter class"),
            ExpectedTensor("segmentation-data", (None, None, 18), "Assignment of daughter slots"),
            ExpectedTensor("segmentation-momentum", (None, None, 4), "Daughter four-momenta"),
            ExpectedTensor("segmentation-full-class", (None, None, None), "Complete-daughter indicator"),
        ),
    }

    def _infer_event_count(self, pdict: dict) -> int | None:
        lengths = [arr.shape[0] for arr in pdict.values() if hasattr(arr, "shape") and arr.shape]
        return min(lengths) if lengths else None

    def _collect_key_rows(self, pdict: dict) -> list[list[str]]:
        rows: list[list[str]] = []
        for key in sorted(pdict):
            arr = pdict[key]
            shape = getattr(arr, "shape", "<no shape>")
            dtype = getattr(arr, "dtype", "<unknown>")
            rows.append([key, str(shape), str(dtype)])
        return rows

    def _collect_task_rows(self, pdict: dict) -> list[list[str]]:
        rows: list[list[str]] = []
        for task, tensors in self.TASK_TENSORS.items():
            available = [t.name for t in tensors if t.name in pdict]
            status = "active" if available else "inactive"
            rows.append([task, status, ", ".join(sorted(available)) or "<none>"])
        return rows

    def _validate_shapes(self, pdict: dict, n_events: int | None) -> list[list[str]]:
        rows: list[list[str]] = []
        expected_map = {t.name: t for t in self.INPUT_TENSORS}
        for tensors in self.TASK_TENSORS.values():
            expected_map.update({t.name: t for t in tensors})

        for name, tensor in expected_map.items():
            if name not in pdict:
                continue
            arr = pdict[name]
            issues = []
            if not hasattr(arr, "shape"):
                issues.append("value has no shape attribute")
            else:
                target = (n_events,) + tensor.expected_shape[1:] if n_events is not None else tensor.expected_shape
                if target[0] is not None and arr.shape[0] != target[0]:
                    issues.append(f"leading dim {arr.shape[0]} != {target[0]}")
                for dim_idx, exp in enumerate(target[1:], start=1):
                    if exp is None:
                        continue
                    if arr.ndim <= dim_idx:
                        issues.append(f"needs >= {dim_idx + 1} dims for {_format_shape(target)}")
                        break
                    if arr.shape[dim_idx] != exp:
                        issues.append(f"dim {dim_idx} = {arr.shape[dim_idx]} (expected {exp})")
            rows.append(
                [
                    name,
                    _format_shape(getattr(arr, "shape", "<no shape>")),
                    _format_shape(tensor.expected_shape if n_events is None else (n_events,) + tensor.expected_shape[1:]),
                    "; ".join(issues) if issues else "✓",
                ]
            )

        if "x" in pdict and "x_mask" in pdict:
            x_shape = getattr(pdict["x"], "shape", None)
            mask_shape = getattr(pdict["x_mask"], "shape", None)
            if x_shape and mask_shape and x_shape[:2] != mask_shape:
                rows.append([
                    "x_mask",
                    _format_shape(mask_shape),
                    _format_shape((x_shape[0], x_shape[1])),
                    "x_mask does not match x batch/particle dimensions",
                ])

        if "segmentation-class" in pdict and "segmentation-full-class" in pdict:
            seg_shape = getattr(pdict["segmentation-class"], "shape", None)
            full_shape = getattr(pdict["segmentation-full-class"], "shape", None)
            if seg_shape and full_shape and seg_shape[:2] != full_shape[:2]:
                rows.append([
                    "segmentation-full-class",
                    _format_shape(full_shape),
                    _format_shape(seg_shape[:2] + full_shape[2:]),
                    "segmentation-class and segmentation-full-class disagree on batch/segments",
                ])

        if "assignments-indices" in pdict and "assignments-mask" in pdict:
            idx_shape = getattr(pdict["assignments-indices"], "shape", None)
            mask_shape = getattr(pdict["assignments-mask"], "shape", None)
            if idx_shape and mask_shape and idx_shape[:2] != mask_shape:
                rows.append([
                    "assignments-mask",
                    _format_shape(mask_shape),
                    _format_shape(idx_shape[:2]),
                    "assignments-mask does not align with assignments-indices",
                ])

        return rows

    def run(self, pdict: dict) -> None:
        if not pdict:
            logger.warning("Received empty dictionary for preprocessing.")
            return

        n_events = self._infer_event_count(pdict)
        logger.info("Sanity check: inferred %s events from leading dimensions.", n_events or "<unknown>")

        key_table = _render_table(
            "Input keys & shapes",
            ["Key", "Shape", "DType"],
            self._collect_key_rows(pdict),
        )
        logger.info("\n%s", key_table)

        task_table = _render_table(
            "Detected tasks",
            ["Task", "Status", "Available tensors"],
            self._collect_task_rows(pdict),
        )
        logger.info("\n%s", task_table)

        validation_rows = self._validate_shapes(pdict, n_events)
        validation_table = _render_table(
            "Shape validation",
            ["Key", "Actual", "Expected", "Notes"],
            validation_rows,
        )
        logger.info("\n%s", validation_table)
