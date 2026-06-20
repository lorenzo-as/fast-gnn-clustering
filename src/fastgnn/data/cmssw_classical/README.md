# CMSSW Classical Trigger-Cell Adapter

This package converts `l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple` trigger-cell
ROOT files into the canonical `fastgnn.data.CaloDataset` parquet layout.

This source is inference-only. The classical trigger-cell ntuple does not provide
the truth links, SimCluster object table, or object properties needed for object
condensation training or labeled evaluation.

Label semantics:

- `UNLABELED_OBJECT_ID = -10` is written for every real trigger cell.
- `0` remains reserved for real detector noise/background in labeled datasets.
- `-1` remains reserved for padding added later by `CaloDataset.as_padded()` or
  batch collation. It is never written for real trigger cells.

The converter requires an explicit `zside` of `1` or `-1`; it does not implement
automatic endcap selection.

Event metadata:

- The top-level converted `event_id` is the canonical processed dataset id. It is
  assigned in output order and remains stable for `CaloDataset` indexing.
- `metadata.source_file` stores the ROOT file path that produced the converted
  event.
- `metadata.source_event_id`, when present, is copied from a top-level scalar
  `event` branch in the ROOT tree for manual lookup in the source file.
- ROOT entry/index metadata is not stored.
