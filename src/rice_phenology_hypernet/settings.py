from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectSettings:
    root: Path
    data_dir: Path
    raw_dir: Path
    boundary_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    features_dir: Path
    models_dir: Path
    eval_dir: Path
    figures_dir: Path
    tables_dir: Path
    docs_dir: Path
    configs_dir: Path


ROOT = Path(__file__).resolve().parents[2]

def build_project_settings(root: Path) -> ProjectSettings:
    return ProjectSettings(
        root=root,
        data_dir=root / "data",
        raw_dir=root / "data" / "raw",
        boundary_dir=root / "data" / "boundary",
        processed_dir=root / "data" / "processed",
        artifacts_dir=root / "artifacts",
        features_dir=root / "artifacts" / "features",
        models_dir=root / "artifacts" / "models",
        eval_dir=root / "artifacts" / "eval",
        figures_dir=root / "artifacts" / "figures",
        tables_dir=root / "artifacts" / "tables",
        docs_dir=root / "docs",
        configs_dir=root / "configs",
    )


SETTINGS = build_project_settings(ROOT)

for path in (
    SETTINGS.processed_dir,
    SETTINGS.features_dir,
    SETTINGS.models_dir,
    SETTINGS.eval_dir,
    SETTINGS.figures_dir,
    SETTINGS.tables_dir,
):
    path.mkdir(parents=True, exist_ok=True)
