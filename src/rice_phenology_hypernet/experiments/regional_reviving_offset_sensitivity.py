from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from rice_phenology_hypernet.experiments.regional_grid_analysis import (
    METRICS_FILENAME,
    REGIONAL_COMPARISON_MAP_FIGURE_FILENAME,
    REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME,
    REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME,
    analyze_regional_grid_projection,
)
from rice_phenology_hypernet.experiments.regional_grid_projection import (
    DEFAULT_REVIVING_OFFSET_DAYS,
    POINT_YEAR_INPUTS_FILENAME,
    YEARLY_PREDICTIONS_FILENAME,
    prepare_regional_grid_inputs,
    run_regional_grid_projection,
)
from rice_phenology_hypernet.runtime import initialize_run, update_run_metadata
from rice_phenology_hypernet.settings import SETTINGS


DEFAULT_SENSITIVITY_PERIOD = "2003_2007"
DEFAULT_REVIVING_OFFSETS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0)
SENSITIVITY_SUBDIR = "regional_reviving_offset_sensitivity"
SUMMARY_METRICS_FILENAME = "regional_grid_reviving_offset_sensitivity_metrics.csv"
SENSITIVITY_METADATA_FILENAME = "regional_reviving_offset_sensitivity_metadata.json"


@dataclass(frozen=True)
class RegionalRevivingOffsetSensitivityOffsetResult:
    reviving_offset_days: float
    inputs_path: Path
    yearly_predictions_path: Path
    metrics_path: Path
    figure_paths: tuple[Path, ...]
    offset_dir: Path
    skipped_existing: bool = False


@dataclass(frozen=True)
class RegionalRevivingOffsetSensitivityResult:
    run_id: str
    period: str
    summary_metrics_path: Path
    metadata_path: Path
    results: tuple[RegionalRevivingOffsetSensitivityOffsetResult, ...]
    metadata: dict[str, Any]


def run_regional_reviving_offset_sensitivity(
    *,
    deployment_run_id: str = "molde4_seed61",
    run_id: str | None = None,
    seed: int = 61,
    offsets: Sequence[float] | None = None,
    period: str = DEFAULT_SENSITIVITY_PERIOD,
    chunk_size: int = 2048,
    num_workers: int | None = None,
    threads_per_worker: int = 1,
    device: str = "cpu",
) -> RegionalRevivingOffsetSensitivityResult:
    if period != DEFAULT_SENSITIVITY_PERIOD:
        raise ValueError(
            f"Regional reviving offset sensitivity is fixed to period={DEFAULT_SENSITIVITY_PERIOD!r}."
        )

    resolved_offsets = _resolve_offsets(offsets)
    run_paths = initialize_run(run_id=run_id)
    target_dir = run_paths.eval_dir / SENSITIVITY_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)

    offset_results: list[RegionalRevivingOffsetSensitivityOffsetResult] = []
    summary_frames: list[pd.DataFrame] = []
    for offset_days in resolved_offsets:
        offset_name = _offset_dir_name(offset_days)
        offset_dir = target_dir / offset_name
        inputs_base_dir = offset_dir / "inputs"
        projection_base_dir = offset_dir / "projection"
        analysis_dir = offset_dir / "analysis"
        figures_dir = offset_dir / "figures"
        metrics_path = analysis_dir / METRICS_FILENAME

        if metrics_path.exists():
            inputs_path = inputs_base_dir / period / POINT_YEAR_INPUTS_FILENAME
            yearly_predictions_path = projection_base_dir / period / YEARLY_PREDICTIONS_FILENAME
            metrics = _read_metrics_with_offset(metrics_path, offset_days=offset_days, period=period)
            summary_frames.append(metrics)
            offset_results.append(
                RegionalRevivingOffsetSensitivityOffsetResult(
                    reviving_offset_days=offset_days,
                    inputs_path=inputs_path,
                    yearly_predictions_path=yearly_predictions_path,
                    metrics_path=metrics_path,
                    figure_paths=_existing_figure_paths(figures_dir),
                    offset_dir=offset_dir,
                    skipped_existing=True,
                )
            )
            continue

        preparation = prepare_regional_grid_inputs(
            period=period,
            output_dir=inputs_base_dir,
            reviving_offset_days=offset_days,
        )
        inputs_path = preparation.point_year_inputs_path
        if inputs_path.name != POINT_YEAR_INPUTS_FILENAME:
            raise ValueError(f"Unexpected regional point-year input path: {inputs_path}")

        projection = run_regional_grid_projection(
            deployment_run_id=deployment_run_id,
            run_id=run_paths.run_id,
            seed=seed,
            period=period,
            input_path=inputs_path,
            chunk_size=chunk_size,
            num_workers=num_workers,
            threads_per_worker=threads_per_worker,
            device=device,
            output_dir=projection_base_dir,
        )
        analysis = analyze_regional_grid_projection(
            period=period,
            yearly_predictions_path=projection.yearly_predictions_path,
            output_dir=analysis_dir,
            figures_dir=figures_dir,
            build_figures=True,
        )

        metrics = _read_metrics_with_offset(analysis.metrics_path, offset_days=offset_days, period=period)
        summary_frames.append(metrics)

        offset_results.append(
            RegionalRevivingOffsetSensitivityOffsetResult(
                reviving_offset_days=offset_days,
                inputs_path=inputs_path,
                yearly_predictions_path=projection.yearly_predictions_path,
                metrics_path=analysis.metrics_path,
                figure_paths=tuple(analysis.figure_paths),
                offset_dir=offset_dir,
                skipped_existing=False,
            )
        )

    summary_metrics = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame(
            columns=["reviving_offset_days", "reviving_rule", "period", "model", "stage", "mae", "rmse", "bias", "r2", "n"]
        )
    )
    summary_metrics_path = target_dir / SUMMARY_METRICS_FILENAME
    summary_metrics.to_csv(summary_metrics_path, index=False)

    metadata_path = target_dir / SENSITIVITY_METADATA_FILENAME
    metadata = {
        "run_id": run_paths.run_id,
        "deployment_run_id": deployment_run_id,
        "seed": int(seed),
        "period": period,
        "reviving_offsets_days": [float(value) for value in resolved_offsets],
        "default_reviving_offset_days": float(DEFAULT_REVIVING_OFFSET_DAYS),
        "output_dir": _display_path(target_dir),
        "summary_metrics": _display_path(summary_metrics_path),
        "offset_outputs": {
            _offset_dir_name(result.reviving_offset_days): {
                "reviving_offset_days": float(result.reviving_offset_days),
                "reviving_rule": _reviving_rule(result.reviving_offset_days),
                "inputs": _display_path(result.inputs_path),
                "yearly_predictions": _display_path(result.yearly_predictions_path),
                "metrics": _display_path(result.metrics_path),
                "figures": [_display_path(path) for path in result.figure_paths],
                "skipped_existing": bool(result.skipped_existing),
            }
            for result in offset_results
        },
        "notes": [
            "Each offset prepares regional inputs, runs regional projection, and runs regional analysis with figures.",
            "No cross-offset difference maps are generated; compare each offset's standard regional figures side by side.",
        ],
    }
    _write_json(metadata_path, metadata)
    update_run_metadata(run_paths.run_id, regional_reviving_offset_sensitivity=metadata)

    return RegionalRevivingOffsetSensitivityResult(
        run_id=run_paths.run_id,
        period=period,
        summary_metrics_path=summary_metrics_path,
        metadata_path=metadata_path,
        results=tuple(offset_results),
        metadata=metadata,
    )


def _resolve_offsets(offsets: Sequence[float] | None) -> tuple[float, ...]:
    source = DEFAULT_REVIVING_OFFSETS if offsets is None else offsets
    resolved = tuple(float(value) for value in source)
    if not resolved:
        raise ValueError("At least one reviving offset is required.")
    labels = [_offset_dir_name(value) for value in resolved]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Reviving offsets produce duplicate output directories: {labels}")
    return resolved


def _format_offset_days(value: float) -> str:
    return f"{float(value):g}"


def _offset_dir_name(value: float) -> str:
    label = _format_offset_days(value).replace("-", "m").replace(".", "p")
    return f"offset_{label}"


def _reviving_rule(offset_days: float) -> str:
    return f"obs_reviving = transplanting_doy + {_format_offset_days(offset_days)}"


def _read_metrics_with_offset(metrics_path: Path, *, offset_days: float, period: str) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    metrics.insert(0, "period", period)
    metrics.insert(0, "reviving_rule", _reviving_rule(offset_days))
    metrics.insert(0, "reviving_offset_days", offset_days)
    return metrics


def _existing_figure_paths(figures_dir: Path) -> tuple[Path, ...]:
    expected = (
        figures_dir / REGIONAL_COMPARISON_MAP_FIGURE_FILENAME,
        figures_dir / REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME,
        figures_dir / REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME,
    )
    existing = tuple(path for path in expected if path.exists())
    return existing if existing else tuple(sorted(figures_dir.glob("*.png")))


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SETTINGS.root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_REVIVING_OFFSETS",
    "DEFAULT_SENSITIVITY_PERIOD",
    "RegionalRevivingOffsetSensitivityOffsetResult",
    "RegionalRevivingOffsetSensitivityResult",
    "SUMMARY_METRICS_FILENAME",
    "run_regional_reviving_offset_sensitivity",
]
