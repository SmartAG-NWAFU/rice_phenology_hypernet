from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from rice_phenology_hypernet.experiments.dvr_core import (
    DEFAULT_WEATHER_FEATURES,
    DVR_STAGE_NAMES,
    PAPER_MODEL_NAMES,
    PHOTO_SENSITIVE_STAGES,
)
from rice_phenology_hypernet.models.physics import trapezoidal_temperature_response
from rice_phenology_hypernet.runtime import initialize_run, update_run_metadata
from rice_phenology_hypernet.settings import SETTINGS


REMOTE_SENSING_GRID_FILENAME = "middle_rice_pixels_0p05deg_median_min10_lat16_35.parquet"
REMOTE_SENSING_GRID_DIR = (
    SETTINGS.data_dir
    / "artifacts"
    / "features"
    / "china_rice_calendar"
)
REGIONAL_PERIODS = ("2003_2007")
DEFAULT_REGIONAL_PERIOD = "2003_2007"
DEFAULT_REVIVING_OFFSET_DAYS = 5.0
REGIONAL_PERIOD_YEAR_RANGES = {
    "2003_2007": (2003, 2007)
}
REGIONAL_PROJECTION_SUBDIR = "regional_grid_projection"
REGIONAL_WEATHER_DIR = SETTINGS.processed_dir / "regional_grid_weather_gee_era5_2003_2022_clean"
REGIONAL_WEATHER_SUMMARY_PATH = REGIONAL_WEATHER_DIR / "regional_weather_point_year_summary.parquet"
REGIONAL_GRID_FEATURE_DIR = SETTINGS.data_dir / "artifacts" / "features" / REGIONAL_PROJECTION_SUBDIR

VALID_POINTS_FILENAME = "regional_grid_valid_points.parquet"
POINT_YEAR_INPUTS_FILENAME = "regional_grid_point_year_inputs.parquet"
EXCLUDED_POINTS_FILENAME = "regional_grid_excluded_points.csv"
INPUT_METADATA_FILENAME = "regional_grid_input_metadata.json"

YEARLY_PREDICTIONS_FILENAME = "regional_grid_yearly_predictions.parquet"
PROJECTION_METADATA_FILENAME = "regional_grid_projection_metadata.json"

PREDICTION_STAGES = tuple(DVR_STAGE_NAMES)
MAIN_ANALYSIS_STAGES = ("heading", "maturity")
WEATHER_SEQUENCE_LIMIT = 180
WEATHER_COLUMNS = (
    "point_id",
    "lat",
    "Date",
    "year",
    "TemAver",
    "TemMin",
    "TemMax",
    "Precipitation",
)
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class RegionalGridPreparationResult:
    period: str
    valid_points_path: Path
    point_year_inputs_path: Path
    excluded_points_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegionalGridPreparationBatchResult:
    results: tuple[RegionalGridPreparationResult, ...]


@dataclass(frozen=True)
class RegionalGridProjectionResult:
    period: str
    yearly_predictions_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegionalGridProjectionBatchResult:
    results: tuple[RegionalGridProjectionResult, ...]


@dataclass(frozen=True)
class RegionalProjectionSpec:
    """Required artifact selection for one regional projection."""

    deployment_run_id: str
    seed: int
    period: str


@dataclass(frozen=True)
class WeatherSequence:
    doy: np.ndarray
    features: np.ndarray
    thermal: np.ndarray
    photo_factor: np.ndarray


@dataclass(frozen=True)
class StageBatch:
    weather_seq: np.ndarray
    base_dvr_seq: np.ndarray
    mask: np.ndarray
    stage_state: np.ndarray | None
    lengths: np.ndarray


@dataclass(frozen=True)
class PreparedDeploymentModel:
    model_name: str
    stage_requirements: dict[str, float]
    artifact_type: str
    materialized: Any

    @property
    def is_process(self) -> bool:
        return self.artifact_type == "process"


class RegionalModelProvider(Protocol):
    """Prepare the four paper models selected by a regional specification."""

    def prepare_models(
        self,
        *,
        spec: RegionalProjectionSpec,
        device: torch.device,
    ) -> list[PreparedDeploymentModel]: ...


@dataclass(frozen=True)
class ProjectionTask:
    year: int
    shard_path: str


@dataclass
class WorkerState:
    period: str
    inputs_by_year: dict[int, pd.DataFrame]
    models: list[PreparedDeploymentModel]
    chunk_size: int


_WORKER_STATE: WorkerState | None = None


def make_point_id(lon: float, lat: float) -> str:
    def _format(value: float) -> str:
        return f"{float(value):.6f}".replace("-", "m").replace(".", "p")

    return f"lon_{_format(lon)}_lat_{_format(lat)}"


def regional_remote_sensing_path(period: str = DEFAULT_REGIONAL_PERIOD) -> Path:
    return REMOTE_SENSING_GRID_DIR / _validate_regional_period(period) / REMOTE_SENSING_GRID_FILENAME


def _validate_regional_period(period: str) -> str:
    if period not in REGIONAL_PERIODS:
        raise ValueError(f"Unknown regional period {period!r}; expected one of {', '.join(REGIONAL_PERIODS)}")
    return period


def _resolve_regional_periods(period: str) -> tuple[str, ...]:
    if period == "all":
        return REGIONAL_PERIODS
    return (_validate_regional_period(period),)


def _regional_period_years(period: str) -> list[int]:
    start_year, end_year = REGIONAL_PERIOD_YEAR_RANGES[_validate_regional_period(period)]
    return list(range(start_year, end_year + 1))


def _period_output_dir(base_dir: Path | str, period: str) -> Path:
    return Path(base_dir) / _validate_regional_period(period)


def _format_offset_days(value: float) -> str:
    return f"{float(value):g}"


def _reviving_rule(offset_days: float | None) -> str:
    if offset_days is None or not np.isfinite(float(offset_days)):
        return "obs_reviving = transplanting_doy + variable_offset"
    return f"obs_reviving = transplanting_doy + {_format_offset_days(float(offset_days))}"


def _infer_reviving_offset_days(frame: pd.DataFrame) -> float | None:
    if "obs_reviving" not in frame.columns or "transplanting_doy" not in frame.columns:
        return None
    obs = pd.to_numeric(frame["obs_reviving"], errors="coerce").to_numpy(dtype=float)
    transplanting = pd.to_numeric(frame["transplanting_doy"], errors="coerce").to_numpy(dtype=float)
    offsets = obs - transplanting
    offsets = offsets[np.isfinite(offsets)]
    if len(offsets) == 0:
        return None
    first = float(offsets[0])
    if np.allclose(offsets, first, atol=1e-6, rtol=0.0):
        return first
    return None


def prepare_regional_grid_inputs(
    *,
    period: str = DEFAULT_REGIONAL_PERIOD,
    remote_sensing_path: Path | str | None = None,
    weather_summary_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    reviving_offset_days: float = DEFAULT_REVIVING_OFFSET_DAYS,
) -> RegionalGridPreparationResult | RegionalGridPreparationBatchResult:
    periods = _resolve_regional_periods(period)
    if len(periods) > 1:
        if remote_sensing_path is not None:
            raise ValueError("--remote-sensing-path can only be used with a single regional period")
        return RegionalGridPreparationBatchResult(
            results=tuple(
                _prepare_regional_grid_inputs_one(
                    period=resolved_period,
                    remote_sensing_path=None,
                    weather_summary_path=weather_summary_path,
                    output_dir=output_dir,
                    reviving_offset_days=reviving_offset_days,
                )
                for resolved_period in periods
            )
        )
    return _prepare_regional_grid_inputs_one(
        period=periods[0],
        remote_sensing_path=remote_sensing_path,
        weather_summary_path=weather_summary_path,
        output_dir=output_dir,
        reviving_offset_days=reviving_offset_days,
    )


def _prepare_regional_grid_inputs_one(
    *,
    period: str,
    remote_sensing_path: Path | str | None = None,
    weather_summary_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    reviving_offset_days: float = DEFAULT_REVIVING_OFFSET_DAYS,
) -> RegionalGridPreparationResult:
    period = _validate_regional_period(period)
    remote_path = Path(remote_sensing_path) if remote_sensing_path is not None else regional_remote_sensing_path(period)
    weather_path = Path(weather_summary_path) if weather_summary_path is not None else REGIONAL_WEATHER_SUMMARY_PATH
    target_dir = _period_output_dir(output_dir or REGIONAL_GRID_FEATURE_DIR, period)
    target_dir.mkdir(parents=True, exist_ok=True)
    period_years = _regional_period_years(period)

    remote = pd.read_parquet(remote_path).copy()
    remote["period"] = period
    remote["point_id"] = [make_point_id(lon, lat) for lon, lat in zip(remote["lon"], remote["lat"])]
    remote = remote.rename(
        columns={
            "heading_doy": "rs_heading_doy",
            "maturity_doy": "rs_maturity_doy",
        }
    )
    reviving_offset = float(reviving_offset_days)
    remote["obs_reviving"] = pd.to_numeric(remote["transplanting_doy"], errors="coerce") + reviving_offset

    weather_summary = pd.read_parquet(weather_path).copy()
    weather_points = weather_summary[["point_id"]].drop_duplicates()
    weather_point_years = weather_summary[["point_id", "year"]].drop_duplicates().copy()
    weather_point_years["year"] = pd.to_numeric(weather_point_years["year"], errors="raise").astype(int)
    weather_point_years = weather_point_years.loc[weather_point_years["year"].isin(period_years)].copy()

    valid_points = remote.merge(weather_points, on="point_id", how="inner")
    excluded_points = remote.loc[~remote["point_id"].isin(valid_points["point_id"])].copy()
    excluded_points["reason"] = "missing_weather"

    point_year_inputs = weather_point_years.merge(
        valid_points[
            [
                "point_id",
                "period",
                "lon",
                "lat",
                "transplanting_doy",
                "obs_reviving",
                "rs_heading_doy",
                "rs_maturity_doy",
            ]
        ],
        on="point_id",
        how="inner",
    )
    point_year_inputs = point_year_inputs.sort_values(["year", "point_id"]).reset_index(drop=True)
    valid_points = valid_points[
        [
            "point_id",
            "period",
            "lon",
            "lat",
            "transplanting_doy",
            "obs_reviving",
            "rs_heading_doy",
            "rs_maturity_doy",
        ]
    ].sort_values("point_id").reset_index(drop=True)
    excluded_points = excluded_points[
        [
            "point_id",
            "period",
            "lon",
            "lat",
            "transplanting_doy",
            "obs_reviving",
            "rs_heading_doy",
            "rs_maturity_doy",
            "reason",
        ]
    ].sort_values("point_id").reset_index(drop=True)

    valid_points_path = target_dir / VALID_POINTS_FILENAME
    point_year_inputs_path = target_dir / POINT_YEAR_INPUTS_FILENAME
    excluded_points_path = target_dir / EXCLUDED_POINTS_FILENAME
    metadata_path = target_dir / INPUT_METADATA_FILENAME

    valid_points.to_parquet(valid_points_path, index=False)
    point_year_inputs.to_parquet(point_year_inputs_path, index=False)
    excluded_points.to_csv(excluded_points_path, index=False)

    short_years = weather_summary.loc[
        pd.to_numeric(weather_summary.get("n_days"), errors="coerce")
        < pd.to_numeric(weather_summary.get("expected_days"), errors="coerce")
    ]
    metadata = {
        "period": period,
        "period_year_range": list(REGIONAL_PERIOD_YEAR_RANGES[period]),
        "remote_sensing_path": _display_path(remote_path),
        "weather_summary_path": _display_path(weather_path),
        "output_dir": _display_path(target_dir),
        "remote_point_count": int(len(remote)),
        "valid_point_count": int(len(valid_points)),
        "excluded_point_count": int(len(excluded_points)),
        "point_year_count": int(len(point_year_inputs)),
        "years": [int(year) for year in sorted(point_year_inputs["year"].unique())],
        "reviving_offset_days": reviving_offset,
        "reviving_rule": _reviving_rule(reviving_offset),
        "short_weather_point_years": int(len(short_years.loc[pd.to_numeric(short_years["year"], errors="coerce").isin(period_years)])) if "year" in short_years.columns else int(len(short_years)),
        "notes": [
            "Remote-sensing transplanting/heading/maturity are period-specific climatological references.",
            "Regional point-years are restricted to the selected remote-sensing period.",
        ],
    }
    _write_json(metadata_path, metadata)

    return RegionalGridPreparationResult(
        period=period,
        valid_points_path=valid_points_path,
        point_year_inputs_path=point_year_inputs_path,
        excluded_points_path=excluded_points_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def run_regional_grid_projection(
    *,
    spec: RegionalProjectionSpec,
    model_provider: RegionalModelProvider,
    run_id: str | None = None,
    input_path: Path | str | None = None,
    weather_dir: Path | str | None = None,
    chunk_size: int = 2048,
    num_workers: int | None = None,
    threads_per_worker: int = 1,
    device: str | torch.device | None = "auto",
    output_dir: Path | str | None = None,
) -> RegionalGridProjectionResult | RegionalGridProjectionBatchResult:
    periods = _resolve_regional_periods(spec.period)
    if output_dir is None:
        run_paths = initialize_run(run_id=run_id)
        effective_run_id = run_paths.run_id
        update_manifest = True
    else:
        run_paths = None
        effective_run_id = run_id
        update_manifest = False

    if len(periods) > 1 and input_path is not None:
        raise ValueError("--input-path can only be used with a single regional period")

    results = []
    for resolved_period in periods:
        base_output_dir = output_dir if output_dir is not None else run_paths.eval_dir / REGIONAL_PROJECTION_SUBDIR  # type: ignore[union-attr]
        target_dir = _period_output_dir(base_output_dir, resolved_period)
        period_spec = RegionalProjectionSpec(
            deployment_run_id=spec.deployment_run_id,
            seed=spec.seed,
            period=resolved_period,
        )
        result = _run_regional_grid_projection_one(
            spec=period_spec,
            model_provider=model_provider,
            run_id=effective_run_id,
            input_path=input_path,
            weather_dir=weather_dir,
            chunk_size=chunk_size,
            num_workers=num_workers,
            threads_per_worker=threads_per_worker,
            device=device,
            output_dir=target_dir,
            update_manifest=False,
        )
        results.append(result)

    if update_manifest and run_paths is not None:
        update_run_metadata(
            run_paths.run_id,
            regional_grid_projection={
                "deployment_run_id": spec.deployment_run_id,
                "seed": int(spec.seed),
                "periods": [result.period for result in results],
                "output_dir": _display_path(run_paths.eval_dir / REGIONAL_PROJECTION_SUBDIR),
                "period_outputs": {
                    result.period: {
                        "yearly_predictions": _display_path(result.yearly_predictions_path),
                        "metadata": _display_path(result.metadata_path),
                        "years": result.metadata["years"],
                        "yearly_prediction_rows": result.metadata["yearly_prediction_rows"],
                        "task_count": result.metadata["task_count"],
                    }
                    for result in results
                },
            },
        )

    if len(results) == 1:
        return results[0]
    return RegionalGridProjectionBatchResult(results=tuple(results))


def _run_regional_grid_projection_one(
    *,
    spec: RegionalProjectionSpec,
    model_provider: RegionalModelProvider,
    run_id: str | None,
    input_path: Path | str | None,
    weather_dir: Path | str | None,
    chunk_size: int,
    num_workers: int | None,
    threads_per_worker: int,
    device: str | torch.device | None,
    output_dir: Path,
    update_manifest: bool,
) -> RegionalGridProjectionResult:
    period = _validate_regional_period(spec.period)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = Path(input_path) if input_path is not None else REGIONAL_GRID_FEATURE_DIR / period / POINT_YEAR_INPUTS_FILENAME
    if not inputs_path.exists():
        prepare_regional_grid_inputs(period=period)
    inputs = pd.read_parquet(inputs_path).copy()
    inputs["year"] = pd.to_numeric(inputs["year"], errors="raise").astype(int)
    inputs = inputs.loc[inputs["year"].isin(_regional_period_years(period))].copy()
    inputs["period"] = period
    inputs = inputs.sort_values(["year", "point_id"]).reset_index(drop=True)
    reviving_offset = _infer_reviving_offset_days(inputs)
    chunk_step = _resolve_chunk_size(chunk_size)
    threads = _resolve_threads_per_worker(threads_per_worker)
    torch_device = _resolve_torch_device(device)
    prepared_models = _prepare_regional_models(
        model_provider,
        spec=spec,
        device=torch_device,
    )

    weather_root = Path(weather_dir) if weather_dir is not None else REGIONAL_WEATHER_DIR
    shard_paths = tuple(sorted(weather_root.glob("regional_weather_daily_clean_shard_*.parquet")))
    if not shard_paths:
        raise FileNotFoundError(f"No regional weather shard files found in {weather_root}")

    model_order = [model.model_name for model in prepared_models]
    tasks = _build_projection_tasks(inputs, shard_paths)
    resolved_num_workers = _resolve_num_workers(num_workers=num_workers, task_count=len(tasks))
    wall_clock_start = time.perf_counter()
    task_frames = _execute_projection_tasks(
        tasks=tasks,
        inputs_path=inputs_path,
        prepared_models=prepared_models,
        period=period,
        chunk_size=chunk_step,
        num_workers=resolved_num_workers,
        threads_per_worker=threads,
    )
    wall_clock_seconds = float(time.perf_counter() - wall_clock_start)
    yearly_predictions = (
        pd.concat([frame for frame in task_frames if not frame.empty], ignore_index=True)
        if task_frames
        else _empty_yearly_predictions()
    )

    yearly_predictions_path = target_dir / YEARLY_PREDICTIONS_FILENAME
    metadata_path = target_dir / PROJECTION_METADATA_FILENAME

    yearly_predictions.to_parquet(yearly_predictions_path, index=False)

    metadata = {
        "deployment_run_id": spec.deployment_run_id,
        "run_id": run_id,
        "seed": int(spec.seed),
        "period": period,
        "period_year_range": list(REGIONAL_PERIOD_YEAR_RANGES[period]),
        "model_order": model_order,
        "inputs_path": _display_path(inputs_path),
        "weather_dir": _display_path(weather_root),
        "output_dir": _display_path(target_dir),
        "years": [int(year) for year in sorted(inputs["year"].unique())],
        "input_point_years": int(len(inputs)),
        "yearly_prediction_rows": int(len(yearly_predictions)),
        "chunk_size": int(chunk_step),
        "num_workers": int(resolved_num_workers),
        "threads_per_worker": int(threads),
        "device": str(torch_device),
        "task_count": int(len(tasks)),
        "task_unit": "year_shard",
        "wall_clock_seconds": wall_clock_seconds,
        "weather_shards": [path.name for path in shard_paths],
        "reviving_offset_days": reviving_offset,
        "reviving_rule": _reviving_rule(reviving_offset),
        "notes": [
            "Regional simulation writes period-specific yearly predictions only.",
            "Regional climatology and heading/maturity metrics are derived in the analysis module.",
            "Remote-sensing transplanting dates are period-specific and define obs_reviving.",
        ],
    }
    _write_json(metadata_path, metadata)

    if update_manifest:
        if run_id is None:
            raise ValueError("run_id is required when updating run metadata")
        update_run_metadata(run_id, regional_grid_projection={period: metadata})

    return RegionalGridProjectionResult(
        period=period,
        yearly_predictions_path=yearly_predictions_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def _execute_projection_tasks(
    *,
    tasks: list[ProjectionTask],
    inputs_path: Path,
    prepared_models: list[PreparedDeploymentModel],
    period: str,
    chunk_size: int,
    num_workers: int,
    threads_per_worker: int,
) -> list[pd.DataFrame]:
    if not tasks:
        return []

    with _thread_environment_override(threads_per_worker):
        if num_workers == 1:
            _initialize_worker_state(
                inputs_path,
                period,
                chunk_size,
                threads_per_worker,
                prepared_models,
            )
            try:
                return [_run_projection_task(task) for task in tasks]
            finally:
                _clear_worker_state()

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_initialize_worker_state,
            initargs=(
                inputs_path,
                period,
                chunk_size,
                threads_per_worker,
                prepared_models,
            ),
        ) as executor:
            return list(executor.map(_run_projection_task, tasks))


def _initialize_worker_state(
    inputs_path: Path | str,
    period: str,
    chunk_size: int,
    threads_per_worker: int,
    prepared_models: list[PreparedDeploymentModel],
) -> None:
    global _WORKER_STATE
    _set_torch_threading(threads_per_worker)
    inputs = pd.read_parquet(Path(inputs_path)).copy()
    inputs["year"] = pd.to_numeric(inputs["year"], errors="raise").astype(int)
    inputs["period"] = _validate_regional_period(period)
    inputs = inputs.sort_values(["year", "point_id"]).reset_index(drop=True)
    inputs_by_year = {
        int(year): frame.reset_index(drop=True)
        for year, frame in inputs.groupby("year", observed=True, sort=True)
    }
    _WORKER_STATE = WorkerState(
        period=_validate_regional_period(period),
        inputs_by_year=inputs_by_year,
        models=prepared_models,
        chunk_size=int(chunk_size),
    )


def _clear_worker_state() -> None:
    global _WORKER_STATE
    _WORKER_STATE = None


def _run_projection_task(task: ProjectionTask) -> pd.DataFrame:
    state = _require_worker_state()
    year_inputs = state.inputs_by_year.get(int(task.year))
    if year_inputs is None or year_inputs.empty:
        return _empty_yearly_predictions()

    shard_path = Path(task.shard_path)
    weather = _read_weather_shard_for_year(shard_path, int(task.year), WEATHER_COLUMNS)
    if weather.empty:
        return _empty_yearly_predictions()

    shard_inputs = year_inputs.loc[year_inputs["point_id"].isin(weather["point_id"].astype(str).unique())].reset_index(drop=True)
    if shard_inputs.empty:
        return _empty_yearly_predictions()

    weather_lookup = _build_weather_lookup(weather)
    buffer = _new_task_buffer()
    chunk_step = int(state.chunk_size)

    for start in range(0, len(shard_inputs), chunk_step):
        chunk = shard_inputs.iloc[start : start + chunk_step].reset_index(drop=True)
        base_columns = _chunk_base_columns(chunk, shard_path.name)
        for prepared_model in state.models:
            predictions = _predict_chunk_for_model(chunk, weather_lookup, prepared_model)
            _append_task_buffer(buffer, base_columns, prepared_model.model_name, predictions)

    return _task_buffer_to_frame(buffer)


def _require_worker_state() -> WorkerState:
    if _WORKER_STATE is None:
        raise RuntimeError("Regional projection worker state is not initialized")
    return _WORKER_STATE


def _new_task_buffer() -> dict[str, list[np.ndarray]]:
    return {column: [] for column in _empty_yearly_predictions().columns}


def _chunk_base_columns(chunk: pd.DataFrame, weather_shard: str) -> dict[str, np.ndarray]:
    size = len(chunk)
    period = chunk["period"].astype(str).to_numpy(dtype=object) if "period" in chunk.columns else np.full(size, DEFAULT_REGIONAL_PERIOD, dtype=object)
    return {
        "point_id": chunk["point_id"].astype(str).to_numpy(dtype=object),
        "period": period,
        "year": pd.to_numeric(chunk["year"], errors="coerce").to_numpy(dtype=np.int32),
        "lon": pd.to_numeric(chunk["lon"], errors="coerce").to_numpy(dtype=np.float64),
        "lat": pd.to_numeric(chunk["lat"], errors="coerce").to_numpy(dtype=np.float64),
        "transplanting_doy": pd.to_numeric(chunk["transplanting_doy"], errors="coerce").to_numpy(dtype=np.float32),
        "obs_reviving": pd.to_numeric(chunk["obs_reviving"], errors="coerce").to_numpy(dtype=np.float32),
        "rs_heading_doy": pd.to_numeric(chunk["rs_heading_doy"], errors="coerce").to_numpy(dtype=np.float32),
        "rs_maturity_doy": pd.to_numeric(chunk["rs_maturity_doy"], errors="coerce").to_numpy(dtype=np.float32),
        "weather_shard": np.full(size, weather_shard, dtype=object),
    }


def _append_task_buffer(
    buffer: dict[str, list[np.ndarray]],
    base_columns: dict[str, np.ndarray],
    model_name: str,
    predictions: np.ndarray,
) -> None:
    size = len(predictions)
    buffer["point_id"].append(base_columns["point_id"])
    buffer["period"].append(base_columns["period"])
    buffer["year"].append(base_columns["year"])
    buffer["lon"].append(base_columns["lon"])
    buffer["lat"].append(base_columns["lat"])
    buffer["transplanting_doy"].append(base_columns["transplanting_doy"])
    buffer["obs_reviving"].append(base_columns["obs_reviving"])
    buffer["rs_heading_doy"].append(base_columns["rs_heading_doy"])
    buffer["rs_maturity_doy"].append(base_columns["rs_maturity_doy"])
    buffer["weather_shard"].append(base_columns["weather_shard"])
    buffer["model"].append(np.full(size, model_name, dtype=object))
    for stage_index, stage_name in enumerate(PREDICTION_STAGES):
        buffer[f"pred_{stage_name}"].append(predictions[:, stage_index].astype(np.float32, copy=False))


def _task_buffer_to_frame(buffer: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    if not buffer["point_id"]:
        return _empty_yearly_predictions()
    data = {column: np.concatenate(chunks) for column, chunks in buffer.items()}
    return pd.DataFrame(data, columns=_empty_yearly_predictions().columns)


def _build_weather_lookup(weather: pd.DataFrame) -> dict[str, WeatherSequence]:
    dates = pd.to_datetime(weather["Date"])
    point_ids = weather["point_id"].astype(str).to_numpy(dtype=object)
    doy = dates.dt.dayofyear.to_numpy(dtype=np.int16)
    lat = pd.to_numeric(weather["lat"], errors="coerce").to_numpy(dtype=np.float32)
    tem_aver = pd.to_numeric(weather["TemAver"], errors="coerce").to_numpy(dtype=np.float32)
    tem_min = pd.to_numeric(weather["TemMin"], errors="coerce").to_numpy(dtype=np.float32)
    tem_max = pd.to_numeric(weather["TemMax"], errors="coerce").to_numpy(dtype=np.float32)
    precipitation = pd.to_numeric(weather["Precipitation"], errors="coerce").to_numpy(dtype=np.float32)

    order = np.lexsort((doy.astype(np.int32), point_ids))
    point_ids = point_ids[order]
    doy = doy[order]
    lat = lat[order]
    tem_aver = tem_aver[order]
    tem_min = tem_min[order]
    tem_max = tem_max[order]
    precipitation = precipitation[order]

    daylength = _vectorized_daylength_from_doy(doy, lat)
    feature_matrix = np.column_stack((tem_aver, tem_min, tem_max, daylength, precipitation)).astype(np.float32, copy=False)
    thermal = trapezoidal_temperature_response(tem_aver).astype(np.float32)
    photo_factor = _photo_response(daylength)

    change_points = np.flatnonzero(point_ids[1:] != point_ids[:-1]) + 1 if len(point_ids) else np.array([], dtype=np.int64)
    starts = np.concatenate(([0], change_points))
    stops = np.concatenate((change_points, [len(point_ids)]))
    lookup: dict[str, WeatherSequence] = {}
    for start, stop in zip(starts, stops):
        lookup[str(point_ids[start])] = WeatherSequence(
            doy=doy[start:stop],
            features=feature_matrix[start:stop],
            thermal=thermal[start:stop],
            photo_factor=photo_factor[start:stop],
        )
    return lookup


def _build_stage_batch(
    *,
    point_ids: list[str],
    weather_lookup: dict[str, WeatherSequence],
    start_doy: np.ndarray,
    transplanting_doy: np.ndarray,
    stage_name: str,
    stage_requirement: float,
    model_name: str,
) -> StageBatch:
    needs_stage_state = model_name not in {"m0_t", "m0_dvr", "m1_v2_dvr"}
    lengths = np.zeros(len(point_ids), dtype=np.int32)
    slices: list[tuple[WeatherSequence, int, int] | None] = []
    for index, point_id in enumerate(point_ids):
        weather = weather_lookup.get(point_id)
        start_value = float(start_doy[index]) if np.isfinite(start_doy[index]) else float("nan")
        if weather is None or not np.isfinite(start_value):
            slices.append(None)
            continue
        start_index = int(np.searchsorted(weather.doy, int(round(start_value)), side="left"))
        stop_index = min(start_index + WEATHER_SEQUENCE_LIMIT, len(weather.doy))
        if stop_index <= start_index:
            slices.append(None)
            continue
        lengths[index] = stop_index - start_index
        slices.append((weather, start_index, stop_index))

    max_len = int(lengths.max()) if len(lengths) else 0
    weather_seq = np.zeros((len(point_ids), max_len, len(DEFAULT_WEATHER_FEATURES)), dtype=np.float32)
    base_dvr_seq = np.zeros((len(point_ids), max_len), dtype=np.float32)
    mask = np.zeros((len(point_ids), max_len), dtype=bool)
    stage_state = np.zeros((len(point_ids), 2), dtype=np.float32) if needs_stage_state else None
    requirement = float(stage_requirement)
    if not np.isfinite(requirement) or requirement <= 0:
        raise ValueError("Regional stage requirements must be finite and positive")

    for index, slice_info in enumerate(slices):
        if slice_info is None:
            continue
        weather, start_index, stop_index = slice_info
        length = int(lengths[index])
        weather_seq[index, :length, :] = weather.features[start_index:stop_index]
        if stage_state is not None:
            stage_state[index, 0] = float(start_doy[index])
            base_transplanting = (
                float(transplanting_doy[index]) if np.isfinite(transplanting_doy[index]) else float(start_doy[index])
            )
            stage_state[index, 1] = float(start_doy[index]) - base_transplanting
        thermal = weather.thermal[start_index:stop_index]
        if model_name == "m0_t":
            base = thermal
        else:
            factor = weather.photo_factor[start_index:stop_index] if stage_name in PHOTO_SENSITIVE_STAGES else 1.0
            base = thermal * factor
        base_dvr_seq[index, :length] = np.asarray(base, dtype=np.float32) / requirement
        mask[index, :length] = True

    return StageBatch(
        weather_seq=weather_seq,
        base_dvr_seq=base_dvr_seq,
        mask=mask,
        stage_state=stage_state,
        lengths=lengths,
    )


def _predict_chunk_for_model(
    chunk: pd.DataFrame,
    weather_lookup: dict[str, WeatherSequence],
    prepared_model: PreparedDeploymentModel,
) -> np.ndarray:
    point_ids = chunk["point_id"].astype(str).tolist()
    start_doy = pd.to_numeric(chunk["obs_reviving"], errors="coerce").to_numpy(dtype=float)
    transplanting_doy = pd.to_numeric(chunk["transplanting_doy"], errors="coerce").to_numpy(dtype=float)
    predictions = np.full((len(chunk), len(PREDICTION_STAGES)), np.nan, dtype=np.float32)
    model_name = prepared_model.model_name

    for stage_index, stage_name in enumerate(PREDICTION_STAGES):
        stage_requirement = prepared_model.stage_requirements[stage_name]
        batch = _build_stage_batch(
            point_ids=point_ids,
            weather_lookup=weather_lookup,
            start_doy=start_doy,
            transplanting_doy=transplanting_doy,
            stage_name=stage_name,
            stage_requirement=stage_requirement,
            model_name=model_name,
        )
        if batch.weather_seq.shape[1] == 0:
            start_doy = np.where(np.isfinite(start_doy), start_doy + 1.0, start_doy)
            continue

        if prepared_model.is_process:
            durations = _predict_duration_from_progress_numpy(batch.base_dvr_seq, batch.mask)
        else:
            durations = _predict_duration_from_progress_torch(
                prepared_model.materialized,
                batch=batch,
                stage_index=stage_index,
                model_name=model_name,
            )

        has_sequence = batch.lengths > 0
        pred_stage = np.where(has_sequence, start_doy + durations.astype(float) - 1.0, np.nan)
        predictions[:, stage_index] = pred_stage.astype(np.float32)
        start_doy = np.where(np.isfinite(pred_stage), np.floor(pred_stage) + 1.0, start_doy + 1.0)

    return predictions


def _predict_duration_from_progress_numpy(base_dvr_seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    progress = np.cumsum(base_dvr_seq * mask.astype(np.float32), axis=1)
    crossed = (progress >= 1.0) & mask
    any_crossed = crossed.any(axis=1)
    first_cross = crossed.argmax(axis=1) + 1
    fallback = mask.sum(axis=1)
    return np.where(any_crossed, first_cross, fallback).astype(np.int32)


def _predict_duration_from_progress_torch(
    model: Any,
    *,
    batch: StageBatch,
    stage_index: int,
    model_name: str,
) -> np.ndarray:
    assert isinstance(model, torch.nn.Module)
    device = _model_device(model)
    if device.type == "cpu":
        inputs: dict[str, torch.Tensor] = {
            "weather_seq": torch.from_numpy(batch.weather_seq),
            "stage_index": torch.full((len(batch.lengths),), int(stage_index), dtype=torch.long),
            "base_dvr_seq": torch.from_numpy(batch.base_dvr_seq),
            "mask": torch.from_numpy(batch.mask),
        }
        if model_name != "m1_v2_dvr" and batch.stage_state is not None:
            inputs["stage_state"] = torch.from_numpy(batch.stage_state)
    else:
        inputs = {
            "weather_seq": torch.from_numpy(batch.weather_seq).to(device),
            "stage_index": torch.full((len(batch.lengths),), int(stage_index), dtype=torch.long, device=device),
            "base_dvr_seq": torch.from_numpy(batch.base_dvr_seq).to(device),
            "mask": torch.from_numpy(batch.mask).to(device),
        }
        if model_name != "m1_v2_dvr" and batch.stage_state is not None:
            inputs["stage_state"] = torch.from_numpy(batch.stage_state).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    cum_progress = outputs["cum_progress_seq"]
    mask = inputs["mask"]
    crossed = (cum_progress >= 1.0) & mask
    any_crossed = crossed.any(dim=1)
    first_cross = torch.argmax(crossed.int(), dim=1) + 1
    fallback = mask.sum(dim=1)
    durations = torch.where(any_crossed, first_cross, fallback)
    return durations.detach().cpu().numpy().astype(np.int32)


def _prepare_regional_models(
    model_provider: RegionalModelProvider,
    *,
    spec: RegionalProjectionSpec,
    device: torch.device,
) -> list[PreparedDeploymentModel]:
    prepared = model_provider.prepare_models(spec=spec, device=device)
    names = [model.model_name for model in prepared]
    if len(names) != len(set(names)):
        raise ValueError("The regional model provider returned duplicate model names")
    unexpected = sorted(set(names) - set(PAPER_MODEL_NAMES))
    if unexpected:
        raise ValueError(
            f"The regional model provider returned unsupported models: {unexpected!r}"
        )
    missing = [name for name in PAPER_MODEL_NAMES if name not in names]
    if missing:
        raise ValueError(
            f"The regional model provider did not return all paper models: {missing!r}"
        )
    for model in prepared:
        missing_stages = [
            stage for stage in DVR_STAGE_NAMES if stage not in model.stage_requirements
        ]
        if missing_stages:
            raise ValueError(
                f"Regional model {model.model_name!r} is missing stage requirements: "
                f"{missing_stages!r}"
            )
    by_name = {model.model_name: model for model in prepared}
    return [by_name[name] for name in PAPER_MODEL_NAMES]


def _read_weather_shard_for_year(shard_path: Path, year: int, columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        return pd.read_parquet(shard_path, columns=list(columns), filters=[("year", "==", int(year))])
    except Exception:
        weather = pd.read_parquet(shard_path, columns=list(columns))
        return weather.loc[pd.to_numeric(weather["year"], errors="coerce") == int(year)].reset_index(drop=True)


def _build_projection_tasks(inputs: pd.DataFrame, shard_paths: tuple[Path, ...]) -> list[ProjectionTask]:
    years = [int(year) for year in sorted(pd.to_numeric(inputs["year"], errors="raise").unique())]
    return [
        ProjectionTask(year=year, shard_path=str(shard_path))
        for year in years
        for shard_path in shard_paths
    ]


def _resolve_num_workers(*, num_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if num_workers is None:
        cpu_total = os.cpu_count() or 1
        return max(1, min(24, task_count, max(1, cpu_total // 4)))
    return max(1, min(int(num_workers), task_count))


def _resolve_threads_per_worker(threads_per_worker: int) -> int:
    return max(1, int(threads_per_worker))


def _resolve_chunk_size(chunk_size: int) -> int:
    return max(1, int(chunk_size))


class _thread_environment_override:
    def __init__(self, threads_per_worker: int) -> None:
        self.threads_per_worker = str(max(1, int(threads_per_worker)))
        self.previous: dict[str, str | None] = {}

    def __enter__(self):
        for env_name in THREAD_ENV_VARS:
            self.previous[env_name] = os.environ.get(env_name)
            os.environ[env_name] = self.threads_per_worker
        return self

    def __exit__(self, exc_type, exc, tb):
        for env_name, previous_value in self.previous.items():
            if previous_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous_value
        return False


def _set_torch_threading(threads_per_worker: int) -> None:
    threads = max(1, int(threads_per_worker))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _resolve_torch_device(
    device: str | torch.device | None,
) -> torch.device:
    """Resolve an operational device choice without experiment defaults."""

    if isinstance(device, torch.device):
        return device
    requested = "auto" if device is None else str(device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _vectorized_daylength(dates: pd.Series, latitude: np.ndarray) -> np.ndarray:
    return _vectorized_daylength_from_doy(dates.dt.dayofyear.to_numpy(dtype=np.int16), latitude)


def _vectorized_daylength_from_doy(doy: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    doy = np.asarray(doy, dtype=float)
    phi = np.deg2rad(latitude.astype(float))
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    p = np.deg2rad(-0.833)
    numerator = np.sin(p) - np.sin(phi) * np.sin(delta)
    denominator = np.cos(phi) * np.cos(delta)
    raw = numerator / denominator
    daylength = np.empty_like(raw, dtype=float)
    above = raw > 1.0
    below = raw < -1.0
    middle = ~(above | below)
    daylength[above] = 0.0
    daylength[below] = 24.0
    daylength[middle] = 24.0 * np.arccos(np.clip(raw[middle], -1.0, 1.0)) / np.pi
    return daylength.astype(np.float32)


def _photo_response(daylength: np.ndarray, *, p_crit: float = 12.5, p_sens: float = 0.2) -> np.ndarray:
    factor = np.ones_like(daylength, dtype=np.float32)
    mask = daylength >= p_crit
    factor[mask] = np.clip(1.0 - (daylength[mask] - p_crit) * p_sens, 0.0, 1.0)
    return factor


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SETTINGS.root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _empty_yearly_predictions() -> pd.DataFrame:
    columns = [
        "point_id",
        "period",
        "year",
        "lon",
        "lat",
        "transplanting_doy",
        "obs_reviving",
        "rs_heading_doy",
        "rs_maturity_doy",
        "weather_shard",
        "model",
        *[f"pred_{stage_name}" for stage_name in PREDICTION_STAGES],
    ]
    return pd.DataFrame(columns=columns)


__all__ = [
    "DEFAULT_REGIONAL_PERIOD",
    "DEFAULT_REVIVING_OFFSET_DAYS",
    "POINT_YEAR_INPUTS_FILENAME",
    "PreparedDeploymentModel",
    "PROJECTION_METADATA_FILENAME",
    "REGIONAL_GRID_FEATURE_DIR",
    "REGIONAL_PERIODS",
    "REGIONAL_PERIOD_YEAR_RANGES",
    "REGIONAL_PROJECTION_SUBDIR",
    "REMOTE_SENSING_GRID_DIR",
    "REMOTE_SENSING_GRID_FILENAME",
    "RegionalGridPreparationResult",
    "RegionalGridPreparationBatchResult",
    "RegionalGridProjectionResult",
    "RegionalGridProjectionBatchResult",
    "RegionalModelProvider",
    "RegionalProjectionSpec",
    "YEARLY_PREDICTIONS_FILENAME",
    "make_point_id",
    "prepare_regional_grid_inputs",
    "regional_remote_sensing_path",
    "run_regional_grid_projection",
]
