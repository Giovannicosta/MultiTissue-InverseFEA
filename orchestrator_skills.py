"""Reusable raw-geometry preprocessing skill for the one-tissue expert."""
from geometry_preprocessing import preprocess_geometry


class OneTissueGeometrySkill:
    """Prepare raw (54-column) geometry for the frozen one-tissue MoE."""

    def __call__(self, raw_geometry):
        return preprocess_geometry(raw_geometry)


class GeometryValidationSkill:
    """Reject missing or non-finite raw 54-feature geometry before prediction."""

    def __call__(self, raw_geometry):
        import numpy as np
        import pandas as pd
        from geometry_preprocessing import GEOMETRY_COLS
        if not isinstance(raw_geometry, pd.DataFrame):
            raise TypeError("Raw geometry must be a pandas DataFrame")
        missing = [col for col in GEOMETRY_COLS if col not in raw_geometry.columns]
        if missing:
            raise ValueError(f"Missing geometry columns: {missing}")
        if not np.isfinite(raw_geometry[GEOMETRY_COLS].to_numpy(dtype=float)).all():
            raise ValueError("Geometry contains non-finite values")
        return raw_geometry.copy()
