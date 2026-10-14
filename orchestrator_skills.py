"""Reusable raw-geometry preprocessing skill for the one-tissue expert."""
from geometry_preprocessing import preprocess_geometry


class OneTissueGeometrySkill:
    """Prepare raw (54-column) geometry for the frozen one-tissue MoE."""

    def __call__(self, raw_geometry):
        return preprocess_geometry(raw_geometry)
