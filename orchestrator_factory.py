"""Factory for a reusable MultiTissue InverseFEA orchestrator.

The initial registry contains exactly one trained expert: tissue_count=1,
target=Part1_E. Adding a two-tissue expert requires separately trained
artifacts and an appropriate preprocessing pipeline.
Only load trusted model artifacts; joblib deserialization runs pickle code.
"""
from pathlib import Path
from multi_tissue_orchestrator import MultiTissueOrchestrator
from one_tissue_moe import OneTissueMoE
from one_tissue_multi_target import OneTissueMultiTargetExpert
from orchestrator_skills import GeometryValidationSkill, OneTissueGeometrySkill


def build_orchestrator(one_tissue_artifact_dir):
    """Load the one-tissue MoE and register its validated inference pipeline.

    Accepts an extracted model directory, not the ZIP file itself.
    """
    artifact_dir = Path(one_tissue_artifact_dir)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Extracted one-tissue model directory not found: {artifact_dir}"
        )

    if all((artifact_dir / name).is_file() for name in (
        "Part3_E_expert.joblib", "Part11_E_expert.joblib"
    )):
        expert = OneTissueMultiTargetExpert.load(artifact_dir)
        targets = ("Part1_E", "Part3_E", "Part11_E")
    else:
        expert = OneTissueMoE.load(artifact_dir)
        targets = ("Part1_E",)
    orchestrator = MultiTissueOrchestrator()
    orchestrator.register(
        tissue_count=1,
        name="OneTissue_Part1_MoE",
        model=expert,
        targets=targets,
    )
    orchestrator.register_pipeline(
        tissue_count=1,
        skills=[
            ("validation", GeometryValidationSkill()),
            ("preprocessing", OneTissueGeometrySkill()),
        ],
    )
    return orchestrator
