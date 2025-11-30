import logging
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExpectedTensor:
    name: str
    expected_shape: tuple
    description: str
    dtype: str | None = None


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
        ExpectedTensor(
            "num_vectors", (None,), "Total global + sequential objects per event", "float32"
        ),
        ExpectedTensor(
            "num_sequential_vectors", (None,), "Sequential entries per event", "float32"
        ),
        ExpectedTensor("x", (None, 18, 7), "Point-cloud tensor (18 particles × 7 features)", "float32"),
        ExpectedTensor("x_mask", (None, 18), "Mask for point-cloud slots", "bool"),
        ExpectedTensor("conditions", (None, None), "Event-level scalars", "float32"),
        ExpectedTensor("conditions_mask", (None, 1), "Mask for event-level scalars", "bool"),
    )

    TASK_TENSORS: dict[str, tuple[ExpectedTensor, ...]] = {
        "classification": (
            ExpectedTensor("classification", (None,), "Per-event class label", "int64"),
            ExpectedTensor("event_weight", (None,), "Optional event weight", "float32"),
        ),
        "truth_generation": (
            ExpectedTensor("x_invisible", (None, None, None), "Invisible particle features", "float32"),
            ExpectedTensor("x_invisible_mask", (None, None), "Mask for invisible particles", "bool"),
            ExpectedTensor("num_invisible_raw", (None,), "Raw count of invisibles", "int64"),
            ExpectedTensor("num_invisible_valid", (None,), "Valid invisibles after matching", "int64"),
        ),
        "resonance_assignment": (
            ExpectedTensor(
                "assignments-indices", (None, None, None), "Resonance-to-child mapping", "int64"
            ),
            ExpectedTensor("assignments-mask", (None, None), "Resonance validity mask", "bool"),
            ExpectedTensor("assignments-indices-mask", (None, None, None), "Per-child mask", "bool"),
            ExpectedTensor("subprocess_id", (None,), "Integer subprocess label", "int64"),
            ExpectedTensor("process_names", (None,), "String subprocess label", None),
        ),
        "segmentation": (
            ExpectedTensor("segmentation-class", (None, None, None), "One-hot daughter class", "bool"),
            ExpectedTensor("segmentation-data", (None, None, 18), "Assignment of daughter slots", "bool"),
            ExpectedTensor("segmentation-momentum", (None, None, 4), "Daughter four-momenta", "float32"),
            ExpectedTensor("segmentation-full-class", (None, None, None), "Complete-daughter indicator", "bool"),
        ),
    }

    DTYPE_TABLE: dict[str, str | None] = {
        **{t.name: t.dtype for t in INPUT_TENSORS},
        **{t.name: t.dtype for tensors in TASK_TENSORS.values() for t in tensors},
    }

    def _infer_event_count(self, pdict: dict) -> int | None:
        lengths = [arr.shape[0] for arr in pdict.values() if hasattr(arr, "shape") and arr.shape]
        return min(lengths) if lengths else None

    def _infer_assignment_dims(self, global_config) -> tuple[int | None, int | None]:
        if global_config is None:
            return None, None

        resonance_slots: int | None = None
        child_slots: int | None = None

        event_particles = getattr(getattr(global_config, "event_info", None), "event_particles", {}) or {}
        if event_particles:
            resonance_slots = sum((len(particles.names) for particles in event_particles.values()))

        product_particles = getattr(getattr(global_config, "event_info", None), "product_particles", {}) or {}
        if product_particles:
            child_slots = max(
                (len(particle.names) for particles in product_particles.values() for particle in particles.values()),
                default=None,
            )

        return resonance_slots, child_slots

    def _infer_segmentation_dims(self, global_config) -> tuple[int | None, int | None]:
        resonance_slots, child_slots = self._infer_assignment_dims(global_config)

        max_children = child_slots + 1 if child_slots is not None else None
        segmentation_tags = None

        segmentation_indices = getattr(getattr(global_config, "event_info", None), "segmentation_indices", None)
        if segmentation_indices is not None:
            segmentation_tags = len(segmentation_indices)

        return max_children, segmentation_tags

    def _collect_key_rows(self, pdict: dict) -> list[list[str]]:
        rows: list[list[str]] = []
        for key in sorted(pdict):
            arr = pdict[key]
            shape = getattr(arr, "shape", "<no shape>")
            dtype = getattr(arr, "dtype", "<unknown>")
            rows.append([key, str(shape), str(dtype)])
        return rows

    def _expected_dtype(self, name: str) -> str | None:
        return self.DTYPE_TABLE.get(name)

    def _validate_dtypes(self, pdict: dict) -> list[list[str]]:
        rows: list[list[str]] = []
        for name, arr in sorted(pdict.items()):
            expected_dtype = self._expected_dtype(name)
            if expected_dtype is None:
                continue

            actual_dtype = getattr(arr, "dtype", None)
            notes = "✓"

            if actual_dtype is None:
                notes = "value has no dtype attribute"
            elif str(actual_dtype) != expected_dtype:
                notes = f"cast from {actual_dtype}"
                try:
                    pdict[name] = arr.astype(expected_dtype)
                except Exception as exc:  # pragma: no cover - defensive
                    notes = f"failed cast from {actual_dtype}: {exc}"
                    logger.warning("Failed to cast %s to %s: %s", name, expected_dtype, exc)
                else:
                    logger.warning("Casting %s from %s to %s", name, actual_dtype, expected_dtype)

            rows.append([
                name,
                str(actual_dtype) if actual_dtype is not None else "<unknown>",
                expected_dtype,
                notes,
            ])

        return rows

    def _collect_task_rows(self, pdict: dict) -> list[list[str]]:
        rows: list[list[str]] = []
        for task, tensors in self.TASK_TENSORS.items():
            available = [t.name for t in tensors if t.name in pdict]
            status = "active" if available else "inactive"
            rows.append([task, status, ", ".join(sorted(available)) or "<none>"])
        return rows

    def _validate_process_mappings(self, pdict: dict, event_particles: dict) -> list[list[str]]:
        if not event_particles:
            return []

        process_names = pdict.get("process_names")
        subprocess_ids = pdict.get("subprocess_id")
        rows: list[list[str]] = []

        ordered_processes = list(event_particles)
        process_lookup = {name: idx for idx, name in enumerate(ordered_processes)}

        if process_names is not None:
            normalized_names = [name.decode() if isinstance(name, (bytes, bytearray)) else str(name) for name in
                                process_names]
            missing = sorted(set(normalized_names) - set(ordered_processes))
            if missing:
                rows.append([
                    "process_names",
                    _format_shape(getattr(process_names, "shape", "<no shape>")),
                    ", ".join(ordered_processes) or "<none>",
                    f"unknown process names: {', '.join(missing)}",
                ])

        if process_names is not None and subprocess_ids is not None:
            normalized_names = [name.decode() if isinstance(name, (bytes, bytearray)) else str(name) for name in
                                process_names]
            mismatches: list[int] = []
            for idx, (name, subprocess) in enumerate(zip(normalized_names, subprocess_ids)):
                expected = process_lookup.get(name)
                if expected is None or expected != subprocess:
                    mismatches.append(idx)

            if mismatches:
                sample = ", ".join(map(str, mismatches[:3]))
                note = f"mismatched subprocess_id for {len(mismatches)} events"
                if len(mismatches) > 3:
                    note += f" (first indices: {sample})"
                else:
                    note += f" (indices: {sample})"
                rows.append([
                    "subprocess_id",
                    _format_shape(getattr(subprocess_ids, "shape", "<no shape>")),
                    "process order",
                    note,
                ])

        if rows:
            assignment_rows: list[list[str]] = []
            for idx, (process, particles) in enumerate(event_particles.items()):
                names = getattr(particles, "names", None)
                if names is None:
                    try:
                        names = list(particles)
                    except TypeError:
                        names = []
                resonance_list = ", ".join(map(str, names)) or "<none>"
                assignment_rows.append([str(idx), process, resonance_list])

            assignment_table = _render_table(
                "Expected subprocess mapping",
                ["Subprocess ID", "Process name", "Resonances"],
                assignment_rows,
            )
            logger.info("\n%s", assignment_table)

        return rows

    def _validate_shapes(
            self,
            pdict: dict,
            n_events: int | None,
            assignment_dims: tuple[int | None, int | None],
            segmentation_dims: tuple[int | None, int | None],
            event_particles: dict,
    ) -> list[list[str]]:
        rows: list[list[str]] = []
        expected_map = {t.name: t for t in self.INPUT_TENSORS}
        for tensors in self.TASK_TENSORS.values():
            expected_map.update({t.name: t for t in tensors})

        resonance_slots, child_slots = assignment_dims
        if resonance_slots is not None and child_slots is not None:
            expected_map["assignments-indices"] = replace(
                expected_map["assignments-indices"],
                expected_shape=(None, resonance_slots, child_slots),
            )
            expected_map["assignments-indices-mask"] = replace(
                expected_map["assignments-indices-mask"],
                expected_shape=(None, resonance_slots, child_slots),
            )
            expected_map["assignments-mask"] = replace(
                expected_map["assignments-mask"],
                expected_shape=(None, resonance_slots),
            )

        seg_children, seg_resonances = segmentation_dims
        if seg_children is not None and seg_resonances is not None:
            expected_map["segmentation-class"] = replace(
                expected_map["segmentation-class"],
                expected_shape=(None, seg_children, seg_resonances),
            )
            expected_map["segmentation-data"] = replace(
                expected_map["segmentation-data"],
                expected_shape=(None, seg_children, 18),
            )
            expected_map["segmentation-momentum"] = replace(
                expected_map["segmentation-momentum"],
                expected_shape=(None, seg_children, 4),
            )
            expected_map["segmentation-full-class"] = replace(
                expected_map["segmentation-full-class"],
                expected_shape=(None, seg_children, seg_resonances),
            )

        missing_rows: list[list[str]] = []
        first_required = self.INPUT_TENSORS[0].name
        if first_required not in pdict:
            missing_rows.append([
                first_required,
                "<missing>",
                _format_shape(expected_map[first_required].expected_shape),
                "missing required tensor",
            ])
            logger.warning("Missing required tensor '%s' in preprocessing input.", first_required)

        for task, tensors in self.TASK_TENSORS.items():
            present = [tensor for tensor in tensors if tensor.name in pdict]
            if not present:
                continue
            for tensor in tensors:
                if tensor.name in pdict:
                    continue
                missing_rows.append([
                    tensor.name,
                    "<missing>",
                    _format_shape(expected_map[tensor.name].expected_shape),
                    f"{task} task missing required tensor",
                ])
                logger.warning("Task '%s' is active but missing tensor '%s'.", task, tensor.name)

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
                    _format_shape(
                        tensor.expected_shape if n_events is None else (n_events,) + tensor.expected_shape[1:]),
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

            rows.extend(self._validate_process_mappings(pdict, event_particles))

        return missing_rows + rows

    def run(self, pdict: dict, global_config=None) -> None:
        if not pdict:
            logger.warning("Received empty dictionary for preprocessing.")
            return

        n_events = self._infer_event_count(pdict)
        assignment_dims = self._infer_assignment_dims(global_config)
        segmentation_dims = self._infer_segmentation_dims(global_config)
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

        validation_rows = self._validate_shapes(
            pdict,
            n_events,
            assignment_dims=assignment_dims,
            segmentation_dims=segmentation_dims,
            event_particles=getattr(getattr(global_config, "event_info", None), "event_particles", {}) or {},
        )
        validation_table = _render_table(
            "Shape validation",
            ["Key", "Actual", "Expected", "Notes"],
            validation_rows,
        )
        logger.info("\n%s", validation_table)

        dtype_rows = self._validate_dtypes(pdict)
        dtype_table = _render_table(
            "DType validation",
            ["Key", "DType", "Expected", "Notes"],
            dtype_rows,
        )
        logger.info("\n%s", dtype_table)

        invalid_values = self._detect_invalid_values(pdict)
        if invalid_values:
            for issue in invalid_values:
                logger.error(
                    "Invalid values detected in '%s': %s",
                    issue["name"],
                    issue["note"],
                )
                if issue["event_rows"]:
                    event_table = _render_table(
                        f"Invalid entries for {issue['name']}",
                        [
                            "Event index",
                            "Invalid count",
                            "Sample positions",
                        ],
                        issue["event_rows"],
                    )
                    logger.error("\n%s", event_table)

            summary = "; ".join(
                f"{issue['name']} has {issue['note']}" for issue in invalid_values
            )
            raise ValueError(
                "Preprocessing aborted due to non-finite values: " + summary
            )

    def _detect_invalid_values(self, pdict: dict) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for name, arr in pdict.items():
            if not hasattr(arr, "dtype"):
                continue
            if arr.dtype.kind not in {"f", "c"}:
                continue

            invalid_mask = ~np.isfinite(arr)
            if not np.any(invalid_mask):
                continue

            invalid_count = int(np.count_nonzero(invalid_mask))
            total_count = int(arr.size)
            sample_indices = np.argwhere(invalid_mask)
            samples = ", ".join(
                map(lambda idx: str(tuple(idx)), sample_indices[:3])
            )
            if len(sample_indices) > 3:
                samples += "..."

            note = (
                f"{invalid_count}/{total_count} entries are NaN/Inf"
                f" (sample indices: {samples or '<scalar>'})"
            )

            issues.append(
                {
                    "name": name,
                    "note": note,
                    "event_rows": self._summarize_invalid_events(invalid_mask),
                }
            )

        return issues

    def _summarize_invalid_events(
        self,
        invalid_mask: np.ndarray,
        max_events: int = 10,
        samples_per_event: int = 3,
    ) -> list[list[str]]:
        if invalid_mask.ndim == 0:
            return [["<scalar>", "1", "<scalar>"]]

        coordinates = np.argwhere(invalid_mask)
        if coordinates.size == 0:
            return []

        event_indices, counts = np.unique(coordinates[:, 0], return_counts=True)
        rows: list[list[str]] = []
        for event_idx, count in zip(event_indices[:max_events], counts[:max_events]):
            event_coords = coordinates[coordinates[:, 0] == event_idx][:samples_per_event]
            samples = ", ".join(
                "<scalar>"
                if coord.size <= 1
                else str(tuple(int(v) for v in coord[1:]))
                for coord in event_coords
            )
            rows.append([str(int(event_idx)), str(int(count)), samples or "<scalar>"])

        return rows
