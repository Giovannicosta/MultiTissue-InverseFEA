"""Explicit tissue-count routing for MultiTissue InverseFEA.

This first version uses a validated tissue-count input, not a learned
tissue-count classifier. Specialists must expose a predict(geometry) method
that returns one scalar Part1_E estimate per row of *preprocessed* geometry.
Other tissue counts and target moduli are not yet supported.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ExpertRegistration:
    tissue_count: int
    name: str
    model: Any
    targets: tuple = ("Part1_E",)


class MultiTissueOrchestrator:
    def __init__(self):
        self._experts = {}
        self._preprocessors = {}
        self._pipelines = {}

    def register(self, tissue_count, name, model, targets=("Part1_E",)):
        if type(tissue_count) is not int or tissue_count < 1:
            raise ValueError("tissue_count must be a positive integer")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Expert name must be a non-empty string")
        if not callable(getattr(model, "predict", None)):
            raise TypeError("Expert must implement predict(geometry)")
        targets = tuple(targets)
        if not targets or not all(isinstance(t, str) and t for t in targets):
            raise ValueError("targets must contain non-empty target names")
        if tissue_count in self._experts:
            raise ValueError(f"Expert already registered for tissue_count={tissue_count}")
        self._experts[tissue_count] = ExpertRegistration(
            tissue_count=tissue_count, name=name, model=model, targets=targets
        )

    def available_tissue_counts(self):
        return sorted(self._experts)

    def predict(self, geometry, *, tissue_count, target="Part1_E"):
        if type(tissue_count) is not int or tissue_count < 1:
            raise ValueError("tissue_count must be a positive integer")
        if tissue_count not in self._experts:
            raise ValueError(
                f"No trained expert for tissue_count={tissue_count}; "
                f"available={self.available_tissue_counts()}"
            )
        expert = self._experts[tissue_count]
        if target not in expert.targets:
            raise ValueError(
                f"Expert '{expert.name}' does not support {target!r}; "
                f"supported={expert.targets}"
            )
        prediction_fn = getattr(expert.model, "predict_target", None)
        if callable(prediction_fn):
            values = np.asarray(prediction_fn(geometry, target), dtype=float).reshape(-1)
        else:
            values = np.asarray(expert.model.predict(geometry), dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Expert generated non-finite predictions")
        n_rows = len(geometry)
        if len(values) != n_rows:
            raise ValueError(
                f"Expert returned {len(values)} predictions for {n_rows} input rows"
            )
        return {
            "tissue_count": tissue_count,
            "expert": expert.name,
            "target": target,
            "predictions": values,
        }


    def register_preprocessing_skill(self, tissue_count, skill):
        """Register a raw-geometry -> model-geometry transformation."""
        if type(tissue_count) is not int or tissue_count < 1:
            raise ValueError("tissue_count must be a positive integer")
        if not callable(skill):
            raise TypeError("Preprocessing skill must be callable")
        if tissue_count in self._preprocessors:
            raise ValueError(f"Preprocessor already registered for tissue_count={tissue_count}")
        self._preprocessors[tissue_count] = skill

    def predict_raw(self, raw_geometry, *, tissue_count, target="Part1_E"):
        """Run registered preprocessing skill, then route to the trained expert."""
        if tissue_count not in self._experts:
            raise ValueError(
                f"No trained expert for tissue_count={tissue_count}; "
                f"available={self.available_tissue_counts()}"
            )
        if tissue_count not in self._preprocessors:
            raise ValueError(
                f"No preprocessing skill for tissue_count={tissue_count}"
            )
        prepared = self._preprocessors[tissue_count](raw_geometry)
        result = self.predict(prepared, tissue_count=tissue_count, target=target)
        return {**result, "preprocessing_skill": type(self._preprocessors[tissue_count]).__name__}

    def register_pipeline(self, tissue_count, skills):
        """Register ordered (name, callable) preprocessing/validation skills."""
        if type(tissue_count) is not int or tissue_count < 1:
            raise ValueError("tissue_count must be a positive integer")
        if tissue_count in self._pipelines:
            raise ValueError(f"Pipeline already registered for tissue_count={tissue_count}")
        steps = []
        for name, skill in skills:
            if not isinstance(name, str) or not name.strip() or not callable(skill):
                raise ValueError("Each step needs a non-empty name and callable skill")
            steps.append((name, skill))
        if not steps:
            raise ValueError("Pipeline must contain at least one skill")
        if len({name for name, _ in steps}) != len(steps):
            raise ValueError("Pipeline skill names must be unique")
        self._pipelines[tissue_count] = tuple(steps)

    def predict_pipeline(self, raw_geometry, *, tissue_count, target="Part1_E"):
        """Run each registered skill in order, then the selected model."""
        if tissue_count not in self._experts:
            raise ValueError(f"No trained expert for tissue_count={tissue_count}; available={self.available_tissue_counts()}")
        if tissue_count not in self._pipelines:
            raise ValueError(f"No skill pipeline for tissue_count={tissue_count}")
        prepared = raw_geometry
        executed = []
        for name, skill in self._pipelines[tissue_count]:
            prepared = skill(prepared)
            executed.append(name)
        return {**self.predict(prepared, tissue_count=tissue_count, target=target),
                "executed_skills": executed}
