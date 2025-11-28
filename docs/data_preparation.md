# 🧪 Data Preparation & Updated Input Reference

This page documents the **new preprocessing contract** for EveNet. The reference schema in
`share/event_info/pretrain.yaml` is bundled with the repository as an example, but you are free to adapt the feature
lists to match your campaign. The goal of this guide is to help you build the `.npz` files that the EveNet converter
ingests and to clarify how each tensor maps onto the model heads.

> ⚠️ **Ownership reminder:** Users are responsible for generating the `.npz` files. EveNet does not reorder or reshape
> features for you—the arrays must already follow the size and ordering implied by your event-info YAML. The converter
> simply validates the layout and writes the parquet outputs. Keep the YAML and the `.npz` dictionary in sync at all
> times.

- [Config + CLI workflow](#config--cli-workflow)
- [Input tensor dictionary](#input-tensor-dictionary)
- [Supervision targets by head](#supervision-targets-by-head)
- [NPZ → Parquet conversion](#npz--parquet-conversion)
- [Runtime checklist](#runtime-checklist)

---

<a id="config--cli-workflow"></a>

## 🛠️ Config + CLI Workflow

1. **Start from an event-info YAML.** The repository ships an example at `share/event_info/pretrain.yaml`; copy or
   extend it to describe the objects, global variables, and heads you plan to enable. The display names inside the
   `INPUTS` block are just labels used for logging and plotting—what matters is the order, which **must** match the
   tensor layout you write to disk.
2. **Produce an event dictionary.** For every event, assemble a Python dictionary that satisfies the shapes described
   below and append it to the archive you will write to disk. When you call `numpy.savez` (or `savez_compressed`), each
   key becomes an array with leading dimension `N`, the number of events in the file. Masks indicate which padded
   entries are valid and should contribute to the loss.
3. **Run the EveNet converter.** Point `preprocessing/preprocess.py` at your `.npz` bundle and pass the matching YAML so
   the loader can recover feature names, the number of sequential vectors, and the heads you are enabling. The converter
   assumes both artifacts describe the same structure—mismatches will surface as validation errors.
4. **Train or evaluate.** Training configs reference the resulting parquet directory via `platform.data_parquet_dir` and
   reuse the same YAML in `options.Dataset.event_info`.

> ✨ **Normalization note.** The `normalize`, `log_normalize`, and `none` tags in the YAML are metadata only. EveNet
> derives channel-wise statistics during conversion. The sole special case is `normalize_uniform`, which reserves a
> transformation for circular variables (`φ`); the model automatically maps to and from the wrapped representation.

---

<a id="input-tensor-dictionary"></a>

## 📦 Input Tensor Dictionary

Each event is described by the following feature tensors. Shapes are shown with a leading `N` to indicate the number of
stored events in a given `.npz` file. Masks share the same leading dimension as the value they gate.

| Key                      | Shape        | Description                                                                                                                                                                                                                                                                                                                                                     |
|--------------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_vectors`            | `(N,)`       | Total count of global + sequential objects per event.                                                                                                                                                                                                                                                                                                           |
| `num_sequential_vectors` | `(N,)`       | Number of valid sequential entries per event. Mirrors `num_vectors` behaviour.                                                                                                                                                                                                                                                                                  |
| `x`                      | `(N, 18, 7)` | Point-cloud tensor storing exactly **18 slots** with **7 features each**. These dimensions are fixed so datasets can leverage the released pretraining weights. Order the features exactly as your YAML lists them (energy, `pT`, `η`, `φ`, b-tag score, lepton flag, charge in the example). Use padding for missing particles and mask them out via `x_mask`. |
| `x_mask`                 | `(N, 18)`    | Boolean (or `0/1`) mask indicating which particle slots in `x` correspond to real objects. Only entries with mask `1` contribute to losses and metrics.                                                                                                                                                                                                         |
| `conditions`             | `(N, C)`     | Event-level scalars. `C` is the number of global variables you define (10 in the example). You may add or remove variables as long as the order matches your YAML; if you do not supply any conditions, drop the key entirely.                                                                                                                                  |
| `conditions_mask`        | `(N, 1)`     | Mask for `conditions`. Set to `1` when the global features are present. If you omit `conditions`, omit this mask as well.                                                                                                                                                                                                                                       |

---

<a id="supervision-targets-by-head"></a>

## 🎯 Supervision Targets by Head

Only provide the tensors required for the heads you enable in your training YAML. Omit unused targets or set them to
empty arrays so the converter skips unnecessary storage.

### Classification Head

| Key              | Shape  | Meaning                                                                                                                                                           |
|------------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `classification` | `(N,)` | Process label per event. Combine with `event_weight` for weighted cross-entropy when sampling imbalanced campaigns.                                               |
| `event_weight`   | `(N,)` | Optional per-event weight; defaults to `1` if omitted. Populate it alongside `classification` so the converter can broadcast the weights into the parquet shards. |

### TruthGeneration Head

| Key                   | Shape             | Meaning                                                                                                                                                                                                                                                               |
|-----------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `x_invisible`         | `(N, N_nu, F_nu)` | Invisible particle (e.g., neutrino) features. `N_nu` is the **maximum** number of invisible objects you intend to pad, `2` in the example, and `F_nu` is the number of features per invisible. Feature order is defined in your YAML under the TruthGeneration block. |
| `x_invisible_mask`    | `(N, N_nu)`       | Flags which invisible entries are valid.                                                                                                                                                                                                                              |
| `num_invisible_raw`   | `(N,)`            | Count of all invisible objects before quality cuts.                                                                                                                                                                                                                   |
| `num_invisible_valid` | `(N,)`            | Number of invisible objects associated with reconstructed parents.                                                                                                                                                                                                    |

### ReconGeneration Head

ReconGeneration is self-supervised: it perturbs the visible point-cloud channels and learns to denoise them. The target
specification (which channels to regenerate) lives **directly in the YAML** under the ReconGeneration configuration. No
additional tensors beyond the standard inputs are required.

### Resonance Assignment Head

| Key                        | Shape       | Meaning                                                                                                                                                                                                        |
|----------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `assignments-indices`      | `(N, R, D)` | Integer indices mapping each resonance to its reconstructed children. `R` = number of resonances per event; `D` = max number of daughters among all resonances. `-1` for no reconstructed children or padding. |
| `assignments-mask`         | `(N, R)`    | Boolean mask indicating whether **all** children of each resonance are successfully reconstructed.                                                                                                             |
| `assignments-indices-mask` | `(N, R, D)` | Per-daughter mask specifying which child indices are valid. `0` indicates padding for missing daughters.                                                                                                       |
| `subprocess_id`            | `(N,)`      | Integer label of the generating subprocess (Feynman diagram class).                                                                                                                                            |
| `process_names`            | `(N,)`      | String label of each subprocess. Must match the ordering in `event_info.yaml` and align with `subprocess_id`. Used only in preprocessing and producing `normalization.pt`.                                     |

> 📐 **Assignment internals:** During conversion EveNet scans your assignment map to determine `R` and `D`, initialises
> arrays filled with `-1`, and then writes the actual child indices along with boolean masks. The snippet below mirrors
> the loader logic so you can generate matching tensors in your own pipeline:

```python
full_indices = np.full((num_events, n_targets, max_daughters), -1, dtype=int)
full_mask = np.zeros((num_events, n_targets), dtype=bool)
index_mask = np.zeros((num_events, n_targets, max_daughters), dtype=bool)
# Fill with your resonance→daughter mappings; mark valid entries in the masks.
```

### Segmentation Head

#### Index conventions
- S = max number of resonance instances across all processes + 1 null instance (null at index S−1)
- Number of resonance tags + 1 null tag (null at index 0)
- **Tensors are boolean** except momentum.

#### Target Tensors

| Key                       | Shape      | Description                                                                                   |
|---------------------------|------------|-----------------------------------------------------------------------------------------------|
| `segmentation-class`      | (N, S, T)  | One-hot resonance **tag** for each resonance **instance**. Null tag = 0; null instance = S-1. |
| `segmentation-data`       | (N, S, 18) | Mask assigning input particles (18 dims) to each instance. Null instance = all zeros.         |
| `segmentation-momentum`   | (N, S, 4)  | True four-momentum (E,px,py,pz) for each instance; null instance = zeros.                     |
| `segmentation-full-class` | (N, S, T)  | 1 if instance is fully reconstructable for tag `t`; else assigned to null tag.                |

#### Quick Summary
- S = instances, T = classes
- Both include a null entry
- Boolean everywhere except 4-momentum
- Segmentation maps:
- instance → tag (`segmentation-class`)
- instance → particles (`segmentation-data`)
- instance → true 4-vector (`segmentation-momentum`)
- instance → fully-reconstructable flag (`segmentation-full-class`)

### Worked Input Example

```python
import numpy as np

example = {
    "x": np.zeros((N, 18, 7), dtype=np.float32),  # fixed to (18, 7) for pretraining compatibility
    "x_mask": np.zeros((N, 18), dtype=bool),
    # Optional globals — drop both keys if unused
    "conditions": np.zeros((N, C), dtype=np.float32),
    "conditions_mask": np.ones((N, 1), dtype=bool),
    # Classification head (weights default to ones if omitted)
    "classification": np.zeros((N,), dtype=np.int32),
    "event_weight": np.ones((N,), dtype=np.float32),
    # Head-specific entries sized by your resonance/segment definitions
    "assignments-indices": np.full((N, R, D), -1, dtype=int),
    "assignments-mask": np.zeros((N, R), dtype=bool),
    "segmentation-data": np.zeros((N, S, 18), dtype=bool),
    # ... add heads you enabled ...
}
```

Feel free to adjust the head-specific dimensions (`R`, `D`, `S`, `T`) and the number of condition scalars `C` to match
your physics process. The only fixed sizes are the point-cloud slots `(18, 7)` shared across datasets. Keep the YAML and
the `.npz` dictionary in sync so the converter knows how many channels to expect and how to name them.

---

<a id="npz--parquet-conversion"></a>

## 🔄 NPZ → Parquet Conversion

1. **Assemble events** into Python lists and save them with `numpy.savez` (or `savez_compressed`). Each key listed above
   becomes an array inside the archive.
2. **Invoke the converter**:

   ```bash
   python preprocessing/preprocess.py \
     share/event_info/pretrain.yaml \
     --in_npz /path/to/events.npz \
     --store_dir /path/to/output \
     --cpu_max 32
   ```

   The converter reads the YAML to recover feature names, masks, and head activation flags, then emits:

    - `data_*.parquet` containing flattened tensors.
    - `shape_metadata.json` with the original shapes (e.g., `(18, 7)` for `x`).
    - `normalization.pt` with channel-wise statistics and class weights.

3. **Inspect the logs.** The script reports how many particles, invisible objects, and resonances were valid across the
   dataset—helpful when debugging mask alignment.

---

<a id="runtime-checklist"></a>

## ✅ Runtime Checklist

- `platform.data_parquet_dir` points to the folder with the generated parquet shards and `shape_metadata.json`.
- `options.Dataset.event_info` references the same YAML (`share/event_info/pretrain.yaml` or your copy).
- `options.Dataset.normalization_file` matches the `normalization.pt` produced during conversion.
- Only the heads you activated in the training YAML have matching supervision tensors in the parquet files.

With those pieces in place, EveNet will rebuild the full event dictionary on the fly, apply the appropriate circular
normalization for `normalize_uniform` channels, and route each tensor to the corresponding head.

