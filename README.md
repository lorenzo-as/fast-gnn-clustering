# fast-gnn-clustering

Technical reference for the CMSSW calorimeter conversion and canonical dataset
format. Processed datasets created before format version `4` must be regenerated.

## Canonical Parquet Schema

`events.parquet` is an Awkward-compatible ragged table with one row per event:

```text
event_id
hits.{retained physical fields, configured derived features, and CMSSW association diagnostics}
truth.hit_object_id
truth.objects.{ROOT truth fields and derived object properties}
metadata.source
metadata.zside
```

Stored hits are real detector vertices only. `truth.hit_object_id` uses:

| Value | Meaning |
| --- | --- |
| `0` | Real detector hit treated as OC background or noise |
| `> 0` | One-based index into `truth.objects` |
| `-1` | Padding sentinel added only during batching |

CMSSW `cluster0` is the single hard assignment used to construct OC labels.
Object-property aggregation may use a different link policy without changing
the OC label convention.

The format version `4` sidecar files are:

| File | Contents |
| --- | --- |
| `metadata.yaml` | Format version, source files, stored `hit_features`, required fields, preprocessing settings, split metadata, and resolved conversion config |
| `normalization.yaml` | Feature-keyed mean and standard deviation mappings for every stored hit feature, computed from the training split |
| `splits.npz` | Deterministic `train`, `val`, and `test` event indices |

## Hit Features

Conversion always retains `x`, `y`, `z`, `energy`, and `layer`. These physical
fields support truncation, diagnostics, and optional object-property analysis
independently of model inputs. `hit_features` selects additional derived fields
to persist; its default list contains every supported field:

| Field | Definition |
| --- | --- |
| `x`, `y`, `z` | CMSSW rechit Cartesian coordinates |
| `r` | `sqrt(x**2 + y**2)` |
| `eta` | Pseudorapidity derived from `x`, `y`, `z`, preserving the endcap sign |
| `phi` | `atan2(y, x)` |
| `energy` | CMSSW rechit energy |
| `et` | `energy / cosh(eta)` |
| `layer` | CMSSW rechit layer |
| `x_over_z` | `x / z`, or `0` when `z == 0` |
| `y_over_z` | `y / z`, or `0` when `z == 0` |
| `log_energy` | `ln(max(energy, log_floor))` |
| `log_et` | `ln(max(et, log_floor))` |

Add new CMSSW derived hit features as `cached_property` methods on `HitFeatures`
in `src/fastgnn/data/cmssw/hit_features.py`; unselected derived arrays are not
built.

The data config `feature_names` list is the ordered model-input selection. It
must be a subset of materialized parquet hit fields. Normalization is projected
from the stored full feature statistics into this ordered subset. Model configs
do not define a second feature list.

`normalization.yaml` stores statistics by feature name rather than by positional
arrays:

```yaml
x: {mean: 0.1, std: 4.0}
energy: {mean: 1.4, std: 2.2}
```

## Object Properties

Object properties are derived from linked rechits. For a link weight `w_i`:

| Field | Definition |
| --- | --- |
| `sum_energy` | `sum(w_i * energy_i)` |
| `sum_et` | `sum(w_i * et_i)` |
| `x_energy_weighted` | `sum(w_i * energy_i * x_i) / sum(w_i * energy_i)` |
| `y_energy_weighted` | `sum(w_i * energy_i * y_i) / sum(w_i * energy_i)` |
| `z_energy_weighted` | `sum(w_i * energy_i * z_i) / sum(w_i * energy_i)` |
| `eta_energy_weighted` | `sum(w_i * energy_i * eta_i) / sum(w_i * energy_i)` |
| `phi_energy_weighted` | `atan2(sum(w_i * energy_i * sin(phi_i)), sum(w_i * energy_i * cos(phi_i)))` |
| `n_hits` | Number of distinct linked rechits |
| `core_shower_length` | Longest consecutive run of occupied rechit layers |

Sums, counts, and shower length are zero for objects without linked hits.
Weighted coordinates are `NaN`. Duplicate links for the same hit and object
contribute to weighted sums but count as one rechit for `n_hits` and layer
occupancy.

Conversion computes raw fields before configured vertex preprocessing and
stores them with `_raw` suffixes. It computes processed fields without suffixes
after preprocessing only for modes registered as property-capable. Initially,
only `rechits_energy_threshold` is property-capable.

ROOT fields retained in `truth.objects` are `impact_eta`, `impact_phi`,
`impact_energy`, `impact_pt`, `sim_energy`, and `track_pdg_id`.

The public utilities are:

```python
from fastgnn.data.object_properties import (
    compute_object_properties,
    compute_object_properties_from_links,
)

compute_object_properties(
    hits, hit_object_id, *, weights=None, object_ids=None, properties=None
)
compute_object_properties_from_links(
    hits, hit_indices, linked_object_ids, weights, *, object_ids=None, properties=None
)
```

`hit_object_id` is the canonical per-hit hard assignment array and uses `0` for
noise. `linked_object_ids` contains one source-native object ID per aggregation
link. `object_ids` optionally defines the dense output row order. The link-based
form accepts arbitrary linked object IDs, including zero, and is used by the
CMSSW source adapter. By default the utilities compute every property and
require `energy`, `layer`, and either stored or derivable coordinates. Callers
that need only a subset may pass `properties`; omitted outputs are not emitted.

## CMSSW Aggregation Modes

Conversion always stores derived truth object properties.
`object_aggregation_mode` controls the rechit links used to compute them:

| Mode | Links | Weight |
| --- | --- | --- |
| `hard_assigned_full` | `cluster0` | `1` |
| `hard_assigned_fractional` | `cluster0` | `abs(frac0)` |
| `all_fractional` | `cluster0..3` | `abs(frac0..3)` |

`hard_assigned_full` is the default. `all_fractional` loads the additional ROOT
association branches only when selected. In every mode, OC labels remain hard
`cluster0` assignments because the current loss accepts exactly one object ID
per hit.

Truth objects are filtered in this order:

1. Selected endcap using the sign of `impact_eta`, when enabled.
2. ROOT object-level energy using `truth_object_energy_field`, when configured.
3. Processed `sum_energy`, when `truth_min_sum_energy` is configured.
4. Compact surviving objects into one-based OC labels.

Hits assigned to filtered truth objects remain real vertices and become OC
background with `truth.hit_object_id == 0`.

## Conversion Configuration

`configs/convert/cmssw.yaml` defines:

| Key | Meaning |
| --- | --- |
| `input_files` | ROOT files containing an `Events` tree |
| `output_dir` | Processed dataset directory |
| `max_events` | Maximum accepted events, or `null` |
| `overwrite` | Whether an existing `events.parquet` may be replaced |
| `zside` | Selected endcap, `-1` or `1` |
| `hit_features` | Persisted derived hit-feature allowlist; core physical fields are always retained |
| `log_floor` | Positive floor used by logarithmic hit features |
| `preprocessing` | Vertex preprocessing registry key |
| `hit_min_energy` | Rechit threshold for `rechits_energy_threshold`, or `null` |
| `object_aggregation_mode` | Object-property rechit-link policy |
| `truth_min_object_energy` | Optional ROOT object-energy threshold |
| `truth_object_energy_field` | ROOT object field used by the object threshold |
| `truth_min_sum_energy` | Optional processed rechit `sum_energy` threshold |
| `filter_truth_by_zside` | Whether to filter truth objects to `zside` |
| `step_size` | Optional uproot iteration step size |
| `train_frac`, `val_frac` | Split fractions; the remainder is test data |
| `seed` | Deterministic split seed |

The current preprocessing registry also reserves ECON-T modes. Reserved modes
raise `NotImplementedError` until their vertex transforms and object-property
semantics are implemented.

Format version `4` uses energy-weighted truth-object position properties.
It also carries forward the format version `3` removal of the old
`coordinate_system`, `preprocessing_mode`, `min_hit_energy`, and
`truth_object_energy_threshold` aliases. Their functionality is represented by
`hit_features`, `preprocessing`, `hit_min_energy`, and
`truth_min_object_energy`, respectively.

## Data Configuration And Training Contract

`configs/data/cmssw_processed.yaml` defines:

| Key | Meaning |
| --- | --- |
| `name` | Dataset adapter name |
| `dataset_dir` | Processed parquet dataset directory |
| `feature_names` | Ordered model-input features selected from parquet |
| `max_events` | Optional per-split event limit |

Training validates both train and validation datasets before the first batch:

1. Every selected data feature exists in parquet.
2. Every selected feature has stored normalization statistics when normalization is enabled.
3. The constructed model input feature dimension equals `len(data.feature_names)`.
4. Every configured payload target field exists in `truth.objects`.
5. The configured payload target dimension equals `model.output_layout.payload.dim`.

Energy-descending truncation reads the always-retained `hits.energy`; `energy`
does not need to be selected as a model input.

## Payload Regression Training

Payload regression is split between model capacity and training semantics:

| Config | Meaning |
| --- | --- |
| `model.output_layout.payload.dim` | Number of generic per-hit payload output channels appended after beta and cluster-space outputs |
| `training.payload.quantities` | Ordered truth-object quantities used as payload targets |
| `training.payload.huber_delta` | Huber delta used for all payload components |
| `training.loss_weights.L_payload` | Global payload-loss weight, optionally scheduled by epoch |

The model config does not name payload truth fields. It only reserves output
channels. The trainer gathers `truth.objects[field][hit_object_id - 1]` for
every selected signal hit, applies the configured transform, and trains all
signal hits with an object-balanced average. Noise and padding targets are
zero-filled and excluded by the label and mask.

Example:

```yaml
model:
  output_dim: 9          # beta + 3 cluster-space + 5 payload channels
  output_layout:
    payload:
      start: 4
      dim: 5

training:
  payload:
    huber_delta: 1.0
    quantities:
      - {name: log_sum_et, field: sum_et, transform: log, epsilon: 1.0e-6, weight: 1.0}
      - {name: eta, field: eta_energy_weighted, transform: identity, weight: 1.0}
      - {name: phi, field: phi_energy_weighted, transform: sin_cos, weight: 1.0}
      - {name: z, field: z_energy_weighted, transform: scale, scale: 100.0, weight: 1.0}
```

Supported transforms are:

| Transform | Output dimension | Target |
| --- | --- | --- |
| `identity` | 1 | `x` |
| `log` | 1 | `log(x + epsilon)` |
| `scale` | 1 | `x / scale` |
| `sin_cos` | 2 | `[sin(x), cos(x)]` |

The payload loss is:

```text
mean over events:
  mean over truth objects in the event:
    mean over selected signal hits in the object:
      weighted Huber(predicted_payload, transformed_target)
```

Objects with no selected hits after truncation do not contribute. Objects with
selected hits use the stored full-object `truth.objects` target, not a target
recomputed from the truncated hit subset.

## Loss Weight Schedules

Every entry in `training.loss_weights` accepts either a scalar, a compact
piecewise-constant point list, or the full scalar-schedule mapping used by
`qmin_schedule`.

```yaml
loss_weights:
  L_V_attractive: 1.0
  L_beta_noise: [[0, 0.1], [10, 0.05]]
  L_payload:
    type: piecewise
    interpolation: linear
    points:
      - [0, 0.0]
      - [5, 1.0]
```

Point lists must start at epoch `0` and use strictly increasing epoch numbers.
