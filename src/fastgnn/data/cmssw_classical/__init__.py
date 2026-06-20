"""Inference-only adapter for classical CMSSW HGCAL trigger-cell ntuples."""

from fastgnn.data.cmssw_classical.dataset import (
    UNLABELED_OBJECT_ID,
    convert_cmssw_classical_root,
    iter_cmssw_classical_events,
)
from fastgnn.data.cmssw_classical.hit_features import build_current_cmssw_features

__all__ = [
    "UNLABELED_OBJECT_ID",
    "build_current_cmssw_features",
    "convert_cmssw_classical_root",
    "iter_cmssw_classical_events",
]
