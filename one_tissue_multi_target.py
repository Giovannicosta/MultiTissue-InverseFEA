"""Multi-output adapter for the current one-tissue Part1 MoE and two RF experts.

This is a target router inside a single tissue configuration; it does not
infer tissue count. Part3_E and Part11_E were trained/evaluated on clean
geometry, so noisy-input accuracy remains untested.
"""
from pathlib import Path
import joblib
import numpy as np
from one_tissue_moe import OneTissueMoE


class OneTissueMultiTargetExpert:
    supported_targets = ("Part1_E", "Part3_E", "Part11_E")

    def __init__(self, part1_moe, part3_model, part11_model):
        self.models = {
            "Part1_E": part1_moe,
            "Part3_E": part3_model,
            "Part11_E": part11_model,
        }

    @classmethod
    def load(cls, artifact_dir):
        directory = Path(artifact_dir)
        missing = [name for name in ("Part3_E_expert.joblib", "Part11_E_expert.joblib")
                   if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing trained target experts: {missing}")
        return cls(
            OneTissueMoE.load(directory),
            joblib.load(directory / "Part3_E_expert.joblib"),
            joblib.load(directory / "Part11_E_expert.joblib"),
        )

    def predict_target(self, geometry, target):
        if target not in self.models:
            raise ValueError(
                f"Unsupported target {target!r}; available={self.supported_targets}"
            )
        return np.asarray(self.models[target].predict(geometry), dtype=float).reshape(-1)

    def predict(self, geometry):
        """Backward-compatible default returns Part1_E predictions."""
        return self.predict_target(geometry, "Part1_E")
