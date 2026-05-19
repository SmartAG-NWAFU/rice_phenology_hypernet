from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

_scipy_gaussian_kde = None

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
from rice_phenology_hypernet.experiments.runner_dvr import DEPLOYMENT_MODEL_NAMES
from rice_phenology_hypernet.figures import builder as figure_builder
from rice_phenology_hypernet.runtime import require_run, update_run_metadata
from rice_phenology_hypernet.settings import SETTINGS


ANALYSIS_METADATA_FILENAME = "regional_grid_projection_analysis_metadata.json"

CLIMATOLOGY_PREDICTIONS_FILENAME = "regional_grid_climatology_predictions.parquet"
METRICS_FILENAME = "regional_grid_heading_maturity_metrics.csv"
PERIOD_METRICS_FILENAME = "regional_grid_period_metrics.csv"

REGIONAL_PHENOLOGY_COMPARISON_FIGURE_FILENAME = "regional_rs_m1_dvr_con_heading_maturity_comparison.png"
REGIONAL_COMPARISON_MAP_FIGURE_FILENAME = REGIONAL_PHENOLOGY_COMPARISON_FIGURE_FILENAME
REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME = "regional_m1_dvr_con_heading_maturity_residual_diagnostics.png"
REGIONAL_RESIDUAL_DISTRIBUTION_FIGURE_FILENAME = REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME
REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME = "regional_m1_dvr_con_heading_maturity_scatter.png"


@dataclass(frozen=True)
class RegionalGridAnalysisResult:
    period: str
    climatology_predictions_path: Path
    metrics_path: Path
    figure_paths: tuple[Path, ...]
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegionalGridAnalysisBatchResult:
    results: tuple[RegionalGridAnalysisResult, ...]
    period_metrics_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegionalGridFigureResult:
    map_path: Path
    residual_diagnostic_path: Path
    scatter_path: Path

    @property
    def residual_distribution_path(self) -> Path:
        return self.residual_diagnostic_path

    @property
    def paths(self) -> tuple[Path, ...]:
        return (self.map_path, self.residual_diagnostic_path, self.scatter_path)


def analyze_regional_grid_projection(
    *,
    run_id: str | None = None,
    period: str = DEFAULT_REGIONAL_PERIOD,
    input_dir: Path | str | None = None,
    yearly_predictions_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
    build_figures: bool = True,
) -> RegionalGridAnalysisResult | RegionalGridAnalysisBatchResult:
    periods = _resolve_regional_periods(period)
    if len(periods) > 1 and yearly_predictions_path is not None:
        raise ValueError("--yearly-predictions-path can only be used with a single regional period")

    results = []
    for resolved_period in periods:
        results.append(
            _analyze_regional_grid_projection_one(
                run_id=run_id,
                period=resolved_period,
                input_dir=input_dir,
                yearly_predictions_path=yearly_predictions_path,
                output_dir=output_dir,
                figures_dir=figures_dir,
                build_figures=build_figures,
            )
        )

    if len(results) == 1:
        return results[0]

    target_dir, effective_run_id, should_update_manifest = _resolve_batch_analysis_output(
        run_id=run_id,
        input_dir=input_dir,
        output_dir=output_dir,
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
                "climatology_predictions": _display_path(result.climatology_predictions_path),
                "metrics": _display_path(result.metrics_path),
                "metadata": _display_path(result.metadata_path),
                "figures": [_display_path(path) for path in result.figure_paths],
            }
            for result in results
        },
        "notes": [
            "Each period analysis uses predictions simulated with that period's remote-sensing transplanting date.",
            "Each period analysis builds remote-sensing/model maps, residual diagnostics, and paired m1_dvr_con scatter diagnostics by default.",
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
                        "climatology_predictions": _display_path(result.climatology_predictions_path),
                        "metrics": _display_path(result.metrics_path),
                        "figures": [_display_path(path) for path in result.figure_paths],
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
    figures_dir: Path | str | None,
    build_figures: bool,
) -> RegionalGridAnalysisResult:
    period = _validate_regional_period(period)
    source_path, target_dir, target_figures_dir, effective_run_id, should_update_manifest = _resolve_analysis_paths(
        run_id=run_id,
        period=period,
        input_dir=input_dir,
        yearly_predictions_path=yearly_predictions_path,
        output_dir=output_dir,
        figures_dir=figures_dir,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    if build_figures:
        target_figures_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing regional yearly predictions: {source_path}")

    yearly_predictions = pd.read_parquet(source_path).copy()
    yearly_predictions["period"] = period
    model_order = _model_order_from_yearly_predictions(yearly_predictions)
    climatology_predictions = _build_climatology_predictions(yearly_predictions, model_names=model_order)
    metrics = _build_heading_maturity_metrics(climatology_predictions, model_names=model_order)

    climatology_predictions_path = target_dir / CLIMATOLOGY_PREDICTIONS_FILENAME
    metrics_path = target_dir / METRICS_FILENAME
    metadata_path = target_dir / ANALYSIS_METADATA_FILENAME
    climatology_predictions.to_parquet(climatology_predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    figure_paths: tuple[Path, ...] = ()
    if build_figures:
        figure_result = build_regional_grid_figures(
            run_id=effective_run_id,
            period=period,
            climatology_path=climatology_predictions_path,
            output_dir=target_figures_dir,
        )
        figure_paths = figure_result.paths

    metadata = {
        "run_id": effective_run_id,
        "period": period,
        "period_year_range": list(REGIONAL_PERIOD_YEAR_RANGES[period]),
        "yearly_predictions_source": _display_path(source_path),
        "output_dir": _display_path(target_dir),
        "figures_dir": _display_path(target_figures_dir) if build_figures else None,
        "model_order": model_order,
        "yearly_prediction_rows": int(len(yearly_predictions)),
        "climatology_rows": int(len(climatology_predictions)),
        "metrics_rows": int(len(metrics)),
        "climatology_predictions": _display_path(climatology_predictions_path),
        "metrics": _display_path(metrics_path),
        "figures": [_display_path(path) for path in figure_paths],
        "notes": [
            "This analysis step builds period-specific multi-year climatology and heading/maturity metrics.",
            "Figures compare remote-sensing and m1_dvr_con heading/maturity DOY and include paired grid-cell scatter diagnostics when figure generation is enabled.",
        ],
    }
    _write_json(metadata_path, metadata)

    if should_update_manifest:
        update_run_metadata(
            effective_run_id,
            regional_grid_projection_analysis={
                "yearly_predictions_source": _display_path(source_path),
                "output_dir": _display_path(target_dir),
                "figures_dir": _display_path(target_figures_dir) if build_figures else None,
                "model_order": model_order,
                "climatology_predictions": _display_path(climatology_predictions_path),
                "metrics": _display_path(metrics_path),
                "figures": [_display_path(path) for path in figure_paths],
            },
        )

    return RegionalGridAnalysisResult(
        period=period,
        climatology_predictions_path=climatology_predictions_path,
        metrics_path=metrics_path,
        figure_paths=figure_paths,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def build_regional_grid_figures(
    *,
    run_id: str | None = None,
    period: str = DEFAULT_REGIONAL_PERIOD,
    climatology_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> RegionalGridFigureResult:
    period = _validate_regional_period(period)
    if climatology_path is None:
        run_paths = require_run(run_id=run_id)
        source_path = run_paths.eval_dir / REGIONAL_PROJECTION_SUBDIR / period / CLIMATOLOGY_PREDICTIONS_FILENAME
        target_dir = Path(output_dir) if output_dir is not None else run_paths.figures_dir / REGIONAL_PROJECTION_SUBDIR / period
    else:
        source_path = Path(climatology_path)
        target_dir = Path(output_dir) if output_dir is not None else SETTINGS.figures_dir / (run_id or "regional_grid_projection") / period
    target_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing regional climatology predictions: {source_path}")

    climatology = pd.read_parquet(source_path).copy()
    map_path = _build_regional_map_figure(climatology, target_dir)
    residual_diagnostic_path = _build_regional_residual_diagnostic_figure(climatology, target_dir)
    scatter_path = _build_regional_scatter_diagnostic_figure(climatology, target_dir)
    return RegionalGridFigureResult(
        map_path=map_path,
        residual_diagnostic_path=residual_diagnostic_path,
        scatter_path=scatter_path,
    )


def _resolve_analysis_paths(
    *,
    run_id: str | None,
    period: str,
    input_dir: Path | str | None,
    yearly_predictions_path: Path | str | None,
    output_dir: Path | str | None,
    figures_dir: Path | str | None,
) -> tuple[Path, Path, Path, str, bool]:
    if yearly_predictions_path is not None:
        source_path = Path(yearly_predictions_path)
        source_dir = source_path.parent
        effective_run_id = run_id or source_dir.name
        target_dir = Path(output_dir) if output_dir is not None else source_dir
        target_figures_dir = Path(figures_dir) if figures_dir is not None else SETTINGS.figures_dir / effective_run_id / period
        return source_path, target_dir, target_figures_dir, effective_run_id, run_id is not None

    if input_dir is None:
        run_paths = require_run(run_id=run_id)
        source_dir = run_paths.eval_dir / REGIONAL_PROJECTION_SUBDIR / period
        source_path = source_dir / YEARLY_PREDICTIONS_FILENAME
        target_dir = Path(output_dir) / period if output_dir is not None else source_dir
        target_figures_dir = Path(figures_dir) / period if figures_dir is not None else run_paths.figures_dir / REGIONAL_PROJECTION_SUBDIR / period
        return source_path, target_dir, target_figures_dir, run_paths.run_id, True

    source_dir = Path(input_dir) / period
    source_path = source_dir / YEARLY_PREDICTIONS_FILENAME
    effective_run_id = run_id or Path(input_dir).name
    target_dir = Path(output_dir) / period if output_dir is not None else source_dir
    target_figures_dir = Path(figures_dir) / period if figures_dir is not None else SETTINGS.figures_dir / effective_run_id / REGIONAL_PROJECTION_SUBDIR / period
    return source_path, target_dir, target_figures_dir, effective_run_id, run_id is not None


def _resolve_batch_analysis_output(
    *,
    run_id: str | None,
    input_dir: Path | str | None,
    output_dir: Path | str | None,
) -> tuple[Path, str, bool]:
    if input_dir is None:
        run_paths = require_run(run_id=run_id)
        target_dir = Path(output_dir) if output_dir is not None else run_paths.eval_dir / REGIONAL_PROJECTION_SUBDIR
        return target_dir, run_paths.run_id, True
    source_dir = Path(input_dir)
    effective_run_id = run_id or source_dir.name
    target_dir = Path(output_dir) if output_dir is not None else source_dir
    return target_dir, effective_run_id, run_id is not None


def _write_period_metrics(results: list[RegionalGridAnalysisResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for result in results:
        metrics = pd.read_csv(result.metrics_path)
        metrics.insert(0, "period", result.period)
        frames.append(metrics)
    period_metrics = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["period", "model", "stage", "mae", "rmse", "bias", "r2", "n"])
    )
    path = output_dir / PERIOD_METRICS_FILENAME
    period_metrics.to_csv(path, index=False)
    return path


def _model_order_from_yearly_predictions(yearly_predictions: pd.DataFrame) -> list[str]:
    if "model" not in yearly_predictions.columns:
        return []
    present = [str(model_name) for model_name in yearly_predictions["model"].dropna().unique()]
    present_set = set(present)
    ordered = [model_name for model_name in DEPLOYMENT_MODEL_NAMES if model_name in present_set]
    extras = sorted(model_name for model_name in present if model_name not in set(ordered))
    return [*ordered, *extras]


def _build_climatology_predictions(yearly_predictions: pd.DataFrame, *, model_names: list[str]) -> pd.DataFrame:
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
            *[f"{model_name}_{stage_name}_doy" for model_name in model_names for stage_name in PREDICTION_STAGES],
        ]
        return pd.DataFrame(columns=columns)

    aggregated = (
        yearly_predictions.groupby(index_columns + ["model"], as_index=False, observed=True)
        .agg({f"pred_{stage_name}": "mean" for stage_name in PREDICTION_STAGES})
        .sort_values(index_columns + ["model"])
        .reset_index(drop=True)
    )

    wide = aggregated[index_columns].drop_duplicates().reset_index(drop=True)
    for model_name in model_names:
        model_frame = aggregated.loc[aggregated["model"] == model_name].copy()
        model_frame = model_frame[index_columns + [f"pred_{stage_name}" for stage_name in PREDICTION_STAGES]]
        rename_map = {
            f"pred_{stage_name}": f"{model_name}_{stage_name}_doy" for stage_name in PREDICTION_STAGES
        }
        model_frame = model_frame.rename(columns=rename_map)
        wide = wide.merge(model_frame, on=index_columns, how="left")
    return wide.sort_values("point_id").reset_index(drop=True)


def _build_heading_maturity_metrics(climatology_predictions: pd.DataFrame, *, model_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for stage_name in MAIN_ANALYSIS_STAGES:
            obs_column = f"rs_{stage_name}_doy"
            pred_column = f"{model_name}_{stage_name}_doy"
            obs = pd.to_numeric(climatology_predictions.get(obs_column), errors="coerce").to_numpy(dtype=float)
            pred = pd.to_numeric(climatology_predictions.get(pred_column), errors="coerce").to_numpy(dtype=float)
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
    return pd.DataFrame(rows, columns=["model", "stage", "mae", "rmse", "bias", "r2", "n"])


def _build_regional_map_figure(climatology: pd.DataFrame, output_dir: Path) -> Path:
    china, provinces = figure_builder._load_boundary()
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    row_specs = (
        ("rs", "Remote sensing"),
        ("m1_dvr_con", _display_model_name("m1_dvr_con")),
        ("kde", "DOY distribution"),
    )
    required_columns = [
        f"{source_name}_{stage_name}_doy"
        for source_name in ("rs", "m1_dvr_con")
        for stage_name, _ in stage_specs
    ]
    missing_columns = [column for column in required_columns if column not in climatology.columns]
    if missing_columns:
        raise ValueError(f"Missing columns required for regional comparison map: {', '.join(missing_columns)}")
    doy_norm = Normalize(*_shared_value_range(climatology, required_columns))
    period_label = _period_label(climatology)
    lon_values = pd.to_numeric(climatology["lon"], errors="coerce")
    lat_values = pd.to_numeric(climatology["lat"], errors="coerce")

    fig, axes = figure_builder.plt.subplots(
        3,
        2,
        figsize=(7.4, 9.6),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(3, 2)
    xmin, xmax, ymin, ymax = figure_builder.CONFIG.figures.map_extent
    doy_scatter: Any | None = None
    panel_labels = ("a", "b", "c", "d", "e", "f")

    for row_index, (source_name, row_label) in enumerate(row_specs):
        for col_index, (stage_name, stage_label) in enumerate(stage_specs):
            ax = axes[row_index, col_index]
            if source_name == "kde":
                _plot_stage_doy_kde_panel(ax, climatology, stage_name)
                ax.set_xlabel("Day of year (DOY)", fontsize=8.5)
                ax.set_ylabel("Density" if col_index == 0 else "", fontsize=8.5)
                ax.set_title("", fontsize=11, fontweight="bold")
                ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
                ax.grid(color="#D9D9D9", linewidth=0.3, alpha=0.65)
            else:
                china.boundary.plot(ax=ax, color="black", linewidth=0.6)
                provinces.boundary.plot(ax=ax, color="#999999", linewidth=0.3)
                values = pd.to_numeric(climatology[f"{source_name}_{stage_name}_doy"], errors="coerce")
                mask = values.notna() & lon_values.notna() & lat_values.notna()
                scatter = ax.scatter(
                    lon_values.loc[mask],
                    lat_values.loc[mask],
                    c=values.loc[mask],
                    cmap="cividis",
                    norm=doy_norm,
                    s=8,
                    marker="s",
                    linewidths=0.0,
                    rasterized=True,
                )
                doy_scatter = scatter
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("", fontsize=8.5)
                ax.set_ylabel("Latitude" if col_index == 0 else "", fontsize=8.5)
                ax.set_title(stage_label if row_index == 0 else "", fontsize=11, fontweight="bold")
                ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
                ax.grid(color="#D9D9D9", linewidth=0.25, alpha=0.55)
            ax.text(
                0.02,
                0.98,
                f"({panel_labels[row_index * len(stage_specs) + col_index]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
            )
            if col_index == 0:
                ax.annotate(
                    row_label,
                    xy=(-0.23, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9.2,
                    fontweight="bold",
                )

    if doy_scatter is not None:
        colorbar = fig.colorbar(
            doy_scatter,
            ax=axes[:2, :].ravel().tolist(),
            orientation="vertical",
            location="right",
            fraction=0.05,
            pad=0.025,
            label="Day of year (DOY)",
        )
        colorbar.ax.tick_params(labelsize=7.5)
        colorbar.set_label("Day of year (DOY)", fontsize=8.5)

    if period_label:
        fig.suptitle(
            f"{period_label}: remote sensing and {_display_model_name('m1_dvr_con')} phenology comparison",
            fontsize=11.5,
            fontweight="bold",
        )

    return figure_builder._save(fig, REGIONAL_COMPARISON_MAP_FIGURE_FILENAME, output_dir)


def _build_regional_residual_diagnostic_figure(climatology: pd.DataFrame, output_dir: Path) -> Path:
    china, provinces = figure_builder._load_boundary()
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    residuals_by_stage = {
        stage_name: _stage_residual_values(climatology, stage_name)
        for stage_name, _ in stage_specs
    }
    residual_limit = _residual_limit(climatology, stage_specs)
    residual_norm = TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit)
    x_limit = _residual_distribution_limit(tuple(residuals_by_stage.values()))
    x_grid = np.linspace(-x_limit, x_limit, 500)
    period_label = _period_label(climatology)
    lon_values = pd.to_numeric(climatology["lon"], errors="coerce")
    lat_values = pd.to_numeric(climatology["lat"], errors="coerce")

    fig, axes = figure_builder.plt.subplots(
        2,
        2,
        figsize=(8.4, 6.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(2, 2)
    xmin, xmax, ymin, ymax = figure_builder.CONFIG.figures.map_extent
    residual_scatter: Any | None = None
    panel_labels = ("a", "b", "c", "d")

    for col_index, (stage_name, stage_label) in enumerate(stage_specs):
        ax = axes[0, col_index]
        china.boundary.plot(ax=ax, color="black", linewidth=0.6)
        provinces.boundary.plot(ax=ax, color="#999999", linewidth=0.3)
        values = (
            pd.to_numeric(climatology[f"m1_dvr_con_{stage_name}_doy"], errors="coerce")
            - pd.to_numeric(climatology[f"rs_{stage_name}_doy"], errors="coerce")
        )
        mask = values.notna() & lon_values.notna() & lat_values.notna()
        residual_scatter = ax.scatter(
            lon_values.loc[mask],
            lat_values.loc[mask],
            c=values.loc[mask],
            cmap="RdBu_r",
            norm=residual_norm,
            s=8,
            marker="s",
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(stage_label, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("", fontsize=8.5)
        ax.set_ylabel("Latitude" if col_index == 0 else "", fontsize=8.5)
        ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
        ax.grid(color="#D9D9D9", linewidth=0.25, alpha=0.55)
        _add_panel_label(ax, panel_labels[col_index])

        ax = axes[1, col_index]
        _plot_residual_distribution_panel(
            ax,
            residuals_by_stage[stage_name],
            x_grid=x_grid,
            x_limit=x_limit,
        )
        ax.set_title("", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Prediction - remote sensing (days)", fontsize=9)
        ax.set_ylabel("Density" if col_index == 0 else "", fontsize=9)
        _add_panel_label(ax, panel_labels[col_index + 2])

    if residual_scatter is not None:
        colorbar = fig.colorbar(
            residual_scatter,
            ax=axes[0, :].tolist(),
            orientation="vertical",
            location="right",
            fraction=0.05,
            pad=0.025,
            extend="both",
            label=f"Residual ({_display_model_name('m1_dvr_con')} - remote sensing, days)",
        )
        colorbar.ax.tick_params(labelsize=7.5)
        colorbar.set_label(f"Residual ({_display_model_name('m1_dvr_con')} - remote sensing, days)", fontsize=8.5)

    model_label = _display_model_name("m1_dvr_con")
    title = f"{period_label}: {model_label} residual diagnostics" if period_label else f"{model_label} residual diagnostics"
    fig.suptitle(title, fontsize=11.2, fontweight="bold")
    return figure_builder._save(fig, REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME, output_dir)


def _build_regional_scatter_diagnostic_figure(climatology: pd.DataFrame, output_dir: Path) -> Path:
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    period_label = _period_label(climatology)
    model_label = _display_model_name("m1_dvr_con")

    fig, axes = figure_builder.plt.subplots(
        1,
        2,
        figsize=(8.8, 4.3),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(1, 2)
    panel_labels = ("a", "b")

    for col_index, (stage_name, stage_label) in enumerate(stage_specs):
        ax = axes[0, col_index]
        _plot_stage_scatter_panel(ax, climatology, stage_name, stage_label, model_label)
        _add_panel_label(ax, panel_labels[col_index])

    title = (
        f"{period_label}: {model_label} paired grid-cell diagnostics"
        if period_label
        else f"{model_label} paired grid-cell diagnostics"
    )
    fig.suptitle(title, fontsize=11.2, fontweight="bold")
    return figure_builder._save(fig, REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME, output_dir)


def _display_model_name(model_name: str) -> str:
    return figure_builder.MODEL_DISPLAY.get(model_name, model_name)


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
    )


def _plot_stage_doy_kde_panel(ax: Any, frame: pd.DataFrame, stage_name: str) -> None:
    obs = pd.to_numeric(frame[f"rs_{stage_name}_doy"], errors="coerce").dropna().to_numpy(dtype=float)
    pred = pd.to_numeric(frame[f"m1_dvr_con_{stage_name}_doy"], errors="coerce").dropna().to_numpy(dtype=float)
    x_grid, x_limits = _distribution_grid((obs, pred), step=5.0, pad=5.0)
    series_specs = (
        ("Remote sensing", obs, "#5F6368", 0.18),
        (_display_model_name("m1_dvr_con"), pred, "#2F7D32", 0.22),
    )
    for label, values, color, alpha in series_specs:
        density = _kernel_density(values, x_grid)
        if density is None:
            continue
        ax.fill_between(x_grid, 0.0, density, color=color, alpha=alpha, linewidth=0.0)
        ax.plot(x_grid, density, color=color, linewidth=1.65, label=label)

    obs_mean = _finite_mean(obs)
    pred_mean = _finite_mean(pred)
    if np.isfinite(obs_mean):
        ax.axvline(obs_mean, color="#5F6368", linestyle="--", linewidth=1.15)
    if np.isfinite(pred_mean):
        ax.axvline(pred_mean, color="#2F7D32", linestyle="--", linewidth=1.15)
    ax.text(
        0.97,
        0.96,
        (
            f"RS mean = {_format_days(obs_mean)}\n"
            f"Model mean = {_format_days(pred_mean)}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.88, "pad": 2.4},
    )
    ax.set_xlim(*x_limits)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.88),
        fontsize=7.2,
        frameon=True,
        framealpha=0.82,
    )


def _plot_stage_scatter_panel(
    ax: Any,
    frame: pd.DataFrame,
    stage_name: str,
    stage_label: str,
    model_label: str,
) -> None:
    obs, pred = _stage_pair_values(frame, stage_name)
    lower, upper = _paired_axis_limits(obs, pred)
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.0, label="1:1")

    if len(obs):
        ax.scatter(
            obs,
            pred,
            s=9,
            color="#2F7D32",
            alpha=0.32,
            edgecolors="none",
            rasterized=True,
        )
        if len(obs) >= 2 and not np.isclose(float(np.std(obs)), 0.0):
            slope, intercept = np.polyfit(obs, pred, 1)
            x_values = np.array([lower, upper], dtype=float)
            y_values = slope * x_values + intercept
            ax.plot(x_values, y_values, color="red", linewidth=1.25, label="Linear fit")
    else:
        ax.text(
            0.5,
            0.5,
            "No finite paired values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    metrics = _stage_metric_summary(frame, stage_name)
    ax.text(
        0.04,
        0.88,
        (
            f"MAE = {_format_days(float(metrics['mae']))}\n"
            f"RMSE = {_format_days(float(metrics['rmse']))}\n"
            f"Bias = {_format_signed_days(float(metrics['bias']))}\n"
            f"R2 = {_format_unitless(float(metrics['r2']))}\n"
            f"n = {int(metrics['n'])}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.88, "pad": 2.4},
    )
    ax.set_title(stage_label, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Remote sensing DOY", fontsize=9)
    ax.set_ylabel(f"{model_label} DOY", fontsize=9)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#D9D9D9", linewidth=0.35, alpha=0.65)
    ax.tick_params(labelsize=8, length=2.5, width=0.6)
    ax.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=0.82)


def _plot_residual_distribution_panel(
    ax: Any,
    values: np.ndarray,
    *,
    x_grid: np.ndarray,
    x_limit: float,
) -> None:
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    if len(values) == 0:
        ax.text(
            0.5,
            0.5,
            "No finite residuals",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        bins = _residual_histogram_bins(values)
        p5, median, p95 = np.quantile(values, [0.05, 0.50, 0.95])
        mean = float(np.mean(values))
        ax.axvspan(p5, p95, color="#8FB3D9", alpha=0.18)
        ax.hist(
            values,
            bins=bins,
            density=True,
            color="#9E9E9E",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
        )
        kde_values = _kernel_density(values, x_grid)
        if kde_values is not None:
            ax.plot(x_grid, kde_values, color="#1F4E79", linewidth=1.8)
        ax.axvline(p5, color="#777777", linestyle=":", linewidth=1.0)
        ax.axvline(p95, color="#777777", linestyle=":", linewidth=1.0)
        ax.axvline(median, color="#1F4E79", linestyle="--", linewidth=1.35)
        ax.axvline(mean, color="#B65A2A", linestyle="-.", linewidth=1.25)
        ax.text(
            0.97,
            0.96,
            (
                f"mean = {mean:.1f} d\n"
                f"median = {median:.1f} d\n"
                f"p5-p95 = {p5:.1f} to {p95:.1f} d"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.88, "pad": 2.4},
        )
    ax.set_xlim(-x_limit, x_limit)
    ax.grid(color="#D9D9D9", linewidth=0.35, alpha=0.65)
    ax.tick_params(labelsize=8)


def _stage_metric_summary(frame: pd.DataFrame, stage_name: str) -> dict[str, float | int]:
    obs, pred = _stage_pair_values(frame, stage_name)
    if not len(obs):
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan"), "n": 0}
    diff = pred - obs
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r2": _r2_score(obs, pred),
        "n": int(len(obs)),
    }


def _stage_pair_values(frame: pd.DataFrame, stage_name: str) -> tuple[np.ndarray, np.ndarray]:
    obs = pd.to_numeric(frame[f"rs_{stage_name}_doy"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(frame[f"m1_dvr_con_{stage_name}_doy"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def _paired_axis_limits(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    _, limits = _distribution_grid((obs, pred), step=5.0, pad=5.0)
    return limits


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _format_days(value: float) -> str:
    return f"{value:.1f} d" if np.isfinite(value) else "NA"


def _format_signed_days(value: float) -> str:
    return f"{value:+.1f} d" if np.isfinite(value) else "NA"


def _format_unitless(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "NA"


def _distribution_grid(
    value_groups: tuple[np.ndarray, ...],
    *,
    step: float,
    pad: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    values = [values[np.isfinite(values)] for values in value_groups if len(values)]
    if not values:
        return np.linspace(0.0, 1.0, 100), (0.0, 1.0)
    combined = np.concatenate(values)
    lower = float(np.floor((combined.min() - pad) / step) * step)
    upper = float(np.ceil((combined.max() + pad) / step) * step)
    if not np.isfinite(lower) or not np.isfinite(upper) or np.isclose(lower, upper):
        upper = lower + step
    return np.linspace(lower, upper, 500), (lower, upper)


def _stage_residual_values(frame: pd.DataFrame, stage_name: str) -> np.ndarray:
    pred = pd.to_numeric(frame[f"m1_dvr_con_{stage_name}_doy"], errors="coerce")
    obs = pd.to_numeric(frame[f"rs_{stage_name}_doy"], errors="coerce")
    return (pred - obs).dropna().to_numpy(dtype=float)


def _residual_distribution_limit(residual_groups: tuple[np.ndarray, ...]) -> float:
    values = [np.abs(values) for values in residual_groups if len(values)]
    if not values:
        return 30.0
    quantile = float(np.quantile(np.concatenate(values), 0.995))
    limit = float(np.ceil(quantile / 5.0) * 5.0)
    if not np.isfinite(limit) or limit <= 0.0:
        return 30.0
    return max(30.0, limit)


def _residual_histogram_bins(values: np.ndarray) -> np.ndarray:
    lower = float(np.floor(np.nanmin(values) / 5.0) * 5.0)
    upper = float(np.ceil(np.nanmax(values) / 5.0) * 5.0)
    if not np.isfinite(lower) or not np.isfinite(upper) or np.isclose(lower, upper):
        lower, upper = -5.0, 5.0
    return np.arange(lower, upper + 5.0, 5.0)


def _kernel_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or np.isclose(float(np.nanstd(finite)), 0.0):
        return None
    return _evaluate_kernel_density(finite, x_grid)


def _evaluate_kernel_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if _scipy_gaussian_kde is not None:
        return _scipy_gaussian_kde(values)(x_grid)
    std = float(np.nanstd(values, ddof=1))
    bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max(std, 1.0)
    scaled = (x_grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * scaled**2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))


def _shared_value_range(frame: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    series = [
        pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        for column in columns
        if column in frame.columns
    ]
    if not series:
        return 0.0, 1.0
    values = np.concatenate(series)
    if len(values) == 0:
        return 0.0, 1.0
    lower = float(np.floor(values.min()))
    upper = float(np.ceil(values.max()))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def _residual_limit(frame: pd.DataFrame, stage_specs: tuple[tuple[str, str], ...]) -> float:
    residuals = []
    for stage_name, _ in stage_specs:
        pred = pd.to_numeric(frame[f"m1_dvr_con_{stage_name}_doy"], errors="coerce")
        obs = pd.to_numeric(frame[f"rs_{stage_name}_doy"], errors="coerce")
        residual = (pred - obs).dropna().to_numpy(dtype=float)
        if len(residual):
            residuals.append(residual)
    if not residuals:
        return 1.0
    limit = float(np.ceil(np.quantile(np.abs(np.concatenate(residuals)), 0.99)))
    if not np.isfinite(limit) or np.isclose(limit, 0.0):
        return 1.0
    return limit


def _period_label(frame: pd.DataFrame) -> str:
    if "period" not in frame.columns:
        return ""
    periods = [str(period) for period in frame["period"].dropna().unique()]
    if len(periods) == 1:
        return periods[0]
    if not periods:
        return ""
    return ", ".join(sorted(periods))


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
    "REGIONAL_COMPARISON_MAP_FIGURE_FILENAME",
    "REGIONAL_PHENOLOGY_COMPARISON_FIGURE_FILENAME",
    "REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME",
    "REGIONAL_RESIDUAL_DISTRIBUTION_FIGURE_FILENAME",
    "REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME",
    "RegionalGridAnalysisBatchResult",
    "RegionalGridAnalysisResult",
    "RegionalGridFigureResult",
    "analyze_regional_grid_projection",
    "build_regional_grid_figures",
]
