"""Regional climatology aggregation and heading/maturity metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rice_phenology_hypernet.experiments.dvr_core import PAPER_MODEL_NAMES
from rice_phenology_hypernet.experiments.regional_grid_projection import (
    DEFAULT_REGIONAL_PERIOD,
    MAIN_ANALYSIS_STAGES,
    PREDICTION_STAGES,
    REGIONAL_PERIOD_YEAR_RANGES,
    REGIONAL_PROJECTION_SUBDIR,
    YEARLY_PREDICTIONS_FILENAME,
    _resolve_regional_periods,
    _validate_regional_period,
)
from rice_phenology_hypernet.runtime import require_run, update_run_metadata
from rice_phenology_hypernet.settings import SETTINGS


ANALYSIS_METADATA_FILENAME = "regional_grid_projection_analysis_metadata.json"
CLIMATOLOGY_PREDICTIONS_FILENAME = "regional_grid_climatology_predictions.parquet"
METRICS_FILENAME = "regional_grid_heading_maturity_metrics.csv"
PERIOD_METRICS_FILENAME = "regional_grid_period_metrics.csv"


@dataclass(frozen=True)
class RegionalGridAnalysisResult:
    """Numeric outputs for one regional period."""

    period: str
    climatology_predictions_path: Path
    metrics_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegionalGridAnalysisBatchResult:
    """Numeric outputs spanning multiple regional periods."""

    results: tuple[RegionalGridAnalysisResult, ...]
    period_metrics_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def analyze_regional_grid_projection(
    *,
    run_id: str | None = None,
    period: str = DEFAULT_REGIONAL_PERIOD,
    input_dir: Path | str | None = None,
    yearly_predictions_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> RegionalGridAnalysisResult | RegionalGridAnalysisBatchResult:
    """Build multi-year climatology and metrics from yearly predictions."""

    periods = _resolve_regional_periods(period)
    if len(periods) > 1 and yearly_predictions_path is not None:
        raise ValueError(
            "yearly_predictions_path can only be used with a single regional period"
        )

    results = [
        _analyze_regional_grid_projection_one(
            run_id=run_id,
            period=resolved_period,
            input_dir=input_dir,
            yearly_predictions_path=yearly_predictions_path,
            output_dir=output_dir,
        )
        for resolved_period in periods
    ]
    if len(results) == 1:
        return results[0]

    target_dir, effective_run_id, should_update_manifest = (
        _resolve_batch_analysis_output(
            run_id=run_id,
            input_dir=input_dir,
            output_dir=output_dir,
        )
    )
    period_metrics_path = _write_period_metrics(results, target_dir)
    metadata_path = target_dir / ANALYSIS_METADATA_FILENAME
    metadata = {
        "run_id": effective_run_id,
        "periods": [result.period for result in results],
        "output_dir": _display_path(target_dir),
        "period_metrics": _display_path(period_metrics_path),
        "period_outputs": {
            result.period: {
                "climatology_predictions": _display_path(
                    result.climatology_predictions_path
                ),
                "metrics": _display_path(result.metrics_path),
                "metadata": _display_path(result.metadata_path),
            }
            for result in results
        },
        "notes": [
            "Each period uses predictions generated with its own remote-sensing transplanting date.",
            "Outputs are limited to numeric climatology and heading/maturity metrics.",
        ],
    }
    _write_json(metadata_path, metadata)

    if should_update_manifest:
        update_run_metadata(
            effective_run_id,
            regional_grid_projection_analysis={
                "periods": [result.period for result in results],
                "output_dir": _display_path(target_dir),
                "period_metrics": _display_path(period_metrics_path),
                "period_outputs": {
                    result.period: {
                        "climatology_predictions": _display_path(
                            result.climatology_predictions_path
                        ),
                        "metrics": _display_path(result.metrics_path),
                    }
                    for result in results
                },
            },
        )

    return RegionalGridAnalysisBatchResult(
        results=tuple(results),
        period_metrics_path=period_metrics_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def _analyze_regional_grid_projection_one(
    *,
    run_id: str | None,
    period: str,
    input_dir: Path | str | None,
    yearly_predictions_path: Path | str | None,
    output_dir: Path | str | None,
) -> RegionalGridAnalysisResult:
    period = _validate_regional_period(period)
    source_path, target_dir, effective_run_id, should_update_manifest = (
        _resolve_analysis_paths(
            run_id=run_id,
            period=period,
            input_dir=input_dir,
            yearly_predictions_path=yearly_predictions_path,
            output_dir=output_dir,
        )
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing regional yearly predictions: {source_path}")

    yearly_predictions = pd.read_parquet(source_path).copy()
    yearly_predictions["period"] = period
    model_order = _model_order_from_yearly_predictions(yearly_predictions)
    climatology_predictions = _build_climatology_predictions(
        yearly_predictions,
        model_names=model_order,
    )
    metrics = _build_heading_maturity_metrics(
        climatology_predictions,
        model_names=model_order,
    )

    climatology_predictions_path = target_dir / CLIMATOLOGY_PREDICTIONS_FILENAME
    metrics_path = target_dir / METRICS_FILENAME
    metadata_path = target_dir / ANALYSIS_METADATA_FILENAME
    climatology_predictions.to_parquet(climatology_predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    metadata = {
        "run_id": effective_run_id,
        "period": period,
        "period_year_range": list(REGIONAL_PERIOD_YEAR_RANGES[period]),
        "yearly_predictions_source": _display_path(source_path),
        "output_dir": _display_path(target_dir),
        "model_order": model_order,
        "yearly_prediction_rows": int(len(yearly_predictions)),
        "climatology_rows": int(len(climatology_predictions)),
        "metrics_rows": int(len(metrics)),
        "climatology_predictions": _display_path(climatology_predictions_path),
        "metrics": _display_path(metrics_path),
        "notes": [
            "This step builds period-specific multi-year climatology.",
            "Metrics compare modeled and remote-sensing heading and maturity DOY.",
        ],
    }
    _write_json(metadata_path, metadata)

    if should_update_manifest:
        update_run_metadata(
            effective_run_id,
            regional_grid_projection_analysis={
                "yearly_predictions_source": _display_path(source_path),
                "output_dir": _display_path(target_dir),
                "model_order": model_order,
                "climatology_predictions": _display_path(
                    climatology_predictions_path
                ),
                "metrics": _display_path(metrics_path),
            },
        )

    return RegionalGridAnalysisResult(
        period=period,
        climatology_predictions_path=climatology_predictions_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def _resolve_analysis_paths(
    *,
    run_id: str | None,
    period: str,
    input_dir: Path | str | None,
    yearly_predictions_path: Path | str | None,
    output_dir: Path | str | None,
) -> tuple[Path, Path, str, bool]:
    if yearly_predictions_path is not None:
        source_path = Path(yearly_predictions_path)
        source_dir = source_path.parent
        effective_run_id = run_id or source_dir.name
        target_dir = Path(output_dir) if output_dir is not None else source_dir
        return source_path, target_dir, effective_run_id, run_id is not None

    if input_dir is None:
        run_paths = require_run(run_id=run_id)
        source_dir = run_paths.eval_dir / REGIONAL_PROJECTION_SUBDIR / period
        source_path = source_dir / YEARLY_PREDICTIONS_FILENAME
        target_dir = Path(output_dir) / period if output_dir is not None else source_dir
        return source_path, target_dir, run_paths.run_id, True

    source_dir = Path(input_dir) / period
    source_path = source_dir / YEARLY_PREDICTIONS_FILENAME
    effective_run_id = run_id or Path(input_dir).name
    target_dir = Path(output_dir) / period if output_dir is not None else source_dir
    return source_path, target_dir, effective_run_id, run_id is not None


def _resolve_batch_analysis_output(
    *,
    run_id: str | None,
    input_dir: Path | str | None,
    output_dir: Path | str | None,
) -> tuple[Path, str, bool]:
    if input_dir is None:
        run_paths = require_run(run_id=run_id)
        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else run_paths.eval_dir / REGIONAL_PROJECTION_SUBDIR
        )
        return target_dir, run_paths.run_id, True
    source_dir = Path(input_dir)
    effective_run_id = run_id or source_dir.name
    target_dir = Path(output_dir) if output_dir is not None else source_dir
    return target_dir, effective_run_id, run_id is not None


def _write_period_metrics(
    results: list[RegionalGridAnalysisResult],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for result in results:
        metrics = pd.read_csv(result.metrics_path)
        metrics.insert(0, "period", result.period)
        frames.append(metrics)
    period_metrics = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["period", "model", "stage", "mae", "rmse", "bias", "r2", "n"]
        )
    )
    path = output_dir / PERIOD_METRICS_FILENAME
    period_metrics.to_csv(path, index=False)
    return path


def _model_order_from_yearly_predictions(
    yearly_predictions: pd.DataFrame,
) -> list[str]:
    if "model" not in yearly_predictions.columns:
        return []
    present = [
        str(model_name)
        for model_name in yearly_predictions["model"].dropna().unique()
    ]
    present_set = set(present)
    ordered = [name for name in PAPER_MODEL_NAMES if name in present_set]
    extras = sorted(name for name in present if name not in set(ordered))
    return [*ordered, *extras]


def _build_climatology_predictions(
    yearly_predictions: pd.DataFrame,
    *,
    model_names: list[str],
) -> pd.DataFrame:
    index_columns = [
        "point_id",
        "period",
        "lon",
        "lat",
        "transplanting_doy",
        "obs_reviving",
        "rs_heading_doy",
        "rs_maturity_doy",
    ]
    if yearly_predictions.empty:
        columns = [
            *index_columns,
            *[
                f"{model_name}_{stage_name}_doy"
                for model_name in model_names
                for stage_name in PREDICTION_STAGES
            ],
        ]
        return pd.DataFrame(columns=columns)

    aggregated = (
        yearly_predictions.groupby(
            index_columns + ["model"],
            as_index=False,
            observed=True,
        )
        .agg({f"pred_{stage}": "mean" for stage in PREDICTION_STAGES})
        .sort_values(index_columns + ["model"])
        .reset_index(drop=True)
    )

    wide = aggregated[index_columns].drop_duplicates().reset_index(drop=True)
    for model_name in model_names:
        model_frame = aggregated.loc[aggregated["model"] == model_name].copy()
        prediction_columns = [
            f"pred_{stage_name}" for stage_name in PREDICTION_STAGES
        ]
        model_frame = model_frame[index_columns + prediction_columns]
        model_frame = model_frame.rename(
            columns={
                f"pred_{stage_name}": f"{model_name}_{stage_name}_doy"
                for stage_name in PREDICTION_STAGES
            }
        )
        wide = wide.merge(model_frame, on=index_columns, how="left")
    return wide.sort_values("point_id").reset_index(drop=True)


def _build_heading_maturity_metrics(
    climatology_predictions: pd.DataFrame,
    *,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for stage_name in MAIN_ANALYSIS_STAGES:
            obs_column = f"rs_{stage_name}_doy"
            pred_column = f"{model_name}_{stage_name}_doy"
            obs = pd.to_numeric(
                climatology_predictions.get(obs_column),
                errors="coerce",
            ).to_numpy(dtype=float)
            pred = pd.to_numeric(
                climatology_predictions.get(pred_column),
                errors="coerce",
            ).to_numpy(dtype=float)
            mask = np.isfinite(obs) & np.isfinite(pred)
            if not mask.any():
                rows.append(
                    {
                        "model": model_name,
                        "stage": stage_name,
                        "mae": float("nan"),
                        "rmse": float("nan"),
                        "bias": float("nan"),
                        "r2": float("nan"),
                        "n": 0,
                    }
                )
                continue
            diff = pred[mask] - obs[mask]
            rows.append(
                {
                    "model": model_name,
                    "stage": stage_name,
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                    "bias": float(np.mean(diff)),
                    "r2": _r2_score(obs[mask], pred[mask]),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["model", "stage", "mae", "rmse", "bias", "r2", "n"],
    )


def _r2_score(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2:
        return float("nan")
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    ss_res = float(np.sum((pred - obs) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SETTINGS.root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "ANALYSIS_METADATA_FILENAME",
    "CLIMATOLOGY_PREDICTIONS_FILENAME",
    "METRICS_FILENAME",
    "PERIOD_METRICS_FILENAME",
    "RegionalGridAnalysisBatchResult",
    "RegionalGridAnalysisResult",
    "analyze_regional_grid_projection",
]
