from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from rice_phenology_hypernet.config import get_project_config
from rice_phenology_hypernet.data import RiceDvrStageDataset, load_clean_data
from rice_phenology_hypernet.data.dataset_dvr import (
    DEFAULT_WEATHER_FEATURES,
    DVR_STAGE_NAMES,
    PHOTO_SENSITIVE_STAGES,
    collate_dvr_batches,
    estimate_stage_rate_priors,
    estimate_stage_requirements,
    estimate_stage_requirements_t,
)
from rice_phenology_hypernet.evaluation import calculate_metrics_frame
from rice_phenology_hypernet.features import build_modeling_dataset
from rice_phenology_hypernet.models.dvr_loss import compute_dvr_loss, first_crossing_day
from rice_phenology_hypernet.models.m1_dvr_con import M1ConDvrConfig, M1ConDvrModel, compute_m1_dvr_con_loss
from rice_phenology_hypernet.models.m1_v2_dvr import M1V2DvrConfig, M1V2DvrModel
from rice_phenology_hypernet.models.physics import oryza2000_photo_response, trapezoidal_temperature_response
from rice_phenology_hypernet.runtime import initialize_run, register_experiment, update_run_metadata
from rice_phenology_hypernet.settings import SETTINGS

from .dvr_summary import aggregate_dvr_relative_change_summaries, build_dvr_relative_change_summary
from .splits import collect_protocol_audit, sample_random_splits, site_extrapolation_splits, year_extrapolation_splits


PREDICTION_STAGE_NAMES = list(DVR_STAGE_NAMES)
MAIN_DVR_TASKS = ("sample", "site", "year")
PUBLIC_DVR_MODEL_NAMES = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")
DEPLOYMENT_MODEL_NAMES = PUBLIC_DVR_MODEL_NAMES


@dataclass(frozen=True)
class DvrExperimentBundle:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DvrDeploymentArtifact:
    artifact_type: str
    model_name: str
    stage_requirements: dict[str, float]
    model_config: dict[str, Any] | None = None
    model_state_dict: dict[str, Any] | None = None
    background_gate_values: list[float] | None = None
    artifact_dir: Path | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MaterializedProcessModel:
    model_name: str
    stage_requirements: dict[str, float]


def _set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_torch_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        requested = device.type
    else:
        requested = "auto" if device is None else str(device).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type == "cpu":
        return batch
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@contextmanager
def _thread_environment_override(threads_per_worker: int):
    thread_count = max(1, int(threads_per_worker))
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous_env = {key: os.environ.get(key) for key in keys}
    previous_torch_threads = torch.get_num_threads()
    for key in keys:
        os.environ[key] = str(thread_count)
    torch.set_num_threads(thread_count)
    try:
        yield
    finally:
        torch.set_num_threads(previous_torch_threads)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_weather_index(weather_df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    return {
        (int(sid), int(year)): group.sort_values("Date").reset_index(drop=True)
        for (sid, year), group in weather_df.groupby(["SID", "year"])
    }


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(float(value)) else float(fallback)


def _base_dvr_sequence(weather: pd.DataFrame, stage_name: str, stage_requirement: float) -> np.ndarray:
    if stage_requirement <= 0:
        raise ValueError("stage_requirement must be positive")
    thermal = trapezoidal_temperature_response(weather["TemAver"].to_numpy(dtype=float)).astype(np.float32)
    if stage_name in PHOTO_SENSITIVE_STAGES:
        daylength = weather["daylength"].to_numpy(dtype=float)
        factor = np.asarray([oryza2000_photo_response(float(value)) for value in daylength], dtype=np.float32)
    else:
        factor = np.ones_like(thermal, dtype=np.float32)
    return (thermal * factor / float(stage_requirement)).astype(np.float32)


def _temperature_dvr_sequence(weather: pd.DataFrame, stage_requirement: float) -> np.ndarray:
    if stage_requirement <= 0:
        raise ValueError("stage_requirement must be positive")
    thermal = trapezoidal_temperature_response(weather["TemAver"].to_numpy(dtype=float)).astype(np.float32)
    return (thermal / float(stage_requirement)).astype(np.float32)


def _predict_duration_from_progress(progress: np.ndarray | torch.Tensor) -> int:
    if torch.is_tensor(progress):
        values = progress.detach().cpu().numpy()
    else:
        values = np.asarray(progress, dtype=float)
    crossed = np.where(values >= 1.0)[0]
    return int(crossed[0] + 1) if len(crossed) else int(len(values))


def _build_splitter(task: str):
    config = get_project_config().experiment
    if task == "sample":
        return lambda df: sample_random_splits(df, n_splits=config.sample.n_splits, seed=config.sample.seed)
    if task == "site":
        return lambda df: site_extrapolation_splits(df, n_splits=config.site.n_splits)
    if task == "year":
        return lambda df: year_extrapolation_splits(df)
    raise ValueError(f"Unsupported DVR task={task}")


def _split_train_validation(train_df: pd.DataFrame, validation_fraction: float, seed: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train_df) <= 1 or validation_fraction <= 0:
        return train_df.reset_index(drop=True), train_df.iloc[0:0].copy()
    rng = np.random.default_rng(seed if seed is not None else 0)
    indices = np.arange(len(train_df))
    rng.shuffle(indices)
    val_size = max(1, min(len(train_df) - 1, int(round(len(train_df) * validation_fraction))))
    val_idx = np.sort(indices[:val_size])
    train_idx = np.sort(indices[val_size:])
    return train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)


def _site_inner_validation_groups(train_df: pd.DataFrame, n_groups: int) -> list[np.ndarray]:
    site_counts = train_df.groupby("SID").size().sort_values(ascending=False)
    n_groups = max(1, min(int(n_groups), int(len(site_counts))))
    groups: list[dict[str, Any]] = [{"weight": 0, "sids": []} for _ in range(n_groups)]
    for sid, count in site_counts.items():
        target_index = min(range(n_groups), key=lambda idx: (groups[idx]["weight"], len(groups[idx]["sids"]), idx))
        groups[target_index]["sids"].append(int(sid))
        groups[target_index]["weight"] += int(count)
    return [np.asarray(group["sids"], dtype=int) for group in groups if group["sids"]]


def _build_inner_validation_split(
    train_df: pd.DataFrame,
    *,
    task: str,
    outer_fold: int,
    validation_fraction: float,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if task == "site" and train_df["SID"].nunique() > 1:
        groups = _site_inner_validation_groups(train_df, max(2, int(round(1 / max(validation_fraction, 1e-6)))))
        group_index = (int(outer_fold) - 1) % len(groups)
        val_sids = set(groups[group_index].tolist())
        val_mask = train_df["SID"].isin(val_sids)
        if val_mask.any() and (~val_mask).any():
            return (
                train_df.loc[~val_mask].reset_index(drop=True),
                train_df.loc[val_mask].reset_index(drop=True),
                {"inner_validation_strategy": "protocol_aligned", "inner_validation_mode": "site_grouped"},
            )
    if task == "year" and train_df["year"].nunique() > 1:
        years = sorted(int(year) for year in train_df["year"].dropna().unique())
        n_val_years = max(1, min(len(years) - 1, int(round(len(years) * validation_fraction))))
        val_years = set(years[-n_val_years:])
        val_mask = train_df["year"].isin(val_years)
        if val_mask.any() and (~val_mask).any():
            train_core = train_df.loc[~val_mask].copy()
            seen_sites = set(train_core["SID"].tolist())
            val = train_df.loc[val_mask & train_df["SID"].isin(seen_sites)].copy()
            if not val.empty:
                return (
                    train_core.reset_index(drop=True),
                    val.reset_index(drop=True),
                    {"inner_validation_strategy": "protocol_aligned", "inner_validation_mode": "year_window"},
                )
    train_core, val = _split_train_validation(train_df, validation_fraction, seed)
    return (
        train_core,
        val,
        {"inner_validation_strategy": "random_holdout", "inner_validation_mode": "sample_random"},
    )


def _build_deployment_inner_validation_split(
    modeling_df: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    train_core, val = _split_train_validation(modeling_df, validation_fraction, seed)
    if val.empty:
        raise ValueError("Deployment validation split is empty; provide more site-year records.")
    return (
        train_core,
        val,
        {"inner_validation_strategy": "random_holdout", "inner_validation_mode": "sample_random"},
    )


def _m1_v2_dvr_config() -> M1V2DvrConfig:
    cfg = get_project_config().experiment.m1_v2_dvr
    return M1V2DvrConfig(
        input_dim=len(DEFAULT_WEATHER_FEATURES),
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
        modifier_cap=cfg.modifier_cap,
        event_beta=cfg.event_beta,
    )


def _m1_dvr_con_config() -> M1ConDvrConfig:
    cfg = get_project_config().experiment.m1_dvr_con
    return M1ConDvrConfig(
        input_dim=len(DEFAULT_WEATHER_FEATURES),
        state_dim=2,
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
        modifier_cap=cfg.modifier_cap,
        event_beta=cfg.event_beta,
        background_gate_prior=cfg.background_gate_prior,
    )


def _learned_training_config(model_name: str):
    config = get_project_config().experiment
    if model_name == "m1_v2_dvr":
        return config.m1_v2_dvr
    if model_name == "m1_dvr_con":
        return config.m1_dvr_con
    raise ValueError(f"Unsupported learned DVR model={model_name}")


def _new_learned_model(model_name: str) -> torch.nn.Module:
    if model_name == "m1_v2_dvr":
        return M1V2DvrModel(config=_m1_v2_dvr_config())
    if model_name == "m1_dvr_con":
        return M1ConDvrModel(config=_m1_dvr_con_config())
    raise ValueError(f"Unsupported learned DVR model={model_name}")


def _evaluate_loader(model_name: str, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    losses: list[float] = []
    maes: list[float] = []
    model.eval()
    cfg = _learned_training_config(model_name)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            inputs = {
                "weather_seq": batch["weather_seq"],
                "stage_index": batch["stage_index"],
                "base_dvr_seq": batch["base_dvr_seq"],
                "mask": batch["mask"],
            }
            if model_name == "m1_dvr_con":
                inputs["stage_state"] = batch["stage_state"]
            outputs = model(**inputs)
            if model_name == "m1_dvr_con":
                loss, stats = compute_m1_dvr_con_loss(
                    outputs,
                    batch["true_duration"],
                    batch["mask"],
                    model=model,
                    stage_index=batch["stage_index"],
                    event_loss_weight=cfg.event_loss_weight,
                    terminal_loss_weight=cfg.terminal_loss_weight,
                    shrink_loss_weight=cfg.shrink_loss_weight,
                    smooth_loss_weight=cfg.smooth_loss_weight,
                    mean_anchor_loss_weight=cfg.mean_anchor_loss_weight,
                    stage_anchor_multipliers=cfg.stage_anchor_multipliers,
                    stage_terminal_weights=cfg.stage_terminal_weights,
                    stage_shrink_multipliers=cfg.stage_shrink_multipliers,
                    gate_prior_weight=cfg.gate_prior_weight,
                    gate_monotonic_weight=cfg.gate_monotonic_weight,
                )
            else:
                loss, stats = compute_dvr_loss(
                    outputs,
                    batch["true_duration"],
                    batch["mask"],
                    stage_index=batch["stage_index"],
                    event_loss_weight=cfg.event_loss_weight,
                    terminal_loss_weight=cfg.terminal_loss_weight,
                    shrink_loss_weight=cfg.shrink_loss_weight,
                    smooth_loss_weight=cfg.smooth_loss_weight,
                    mean_anchor_loss_weight=cfg.mean_anchor_loss_weight,
                    stage_anchor_multipliers=cfg.stage_anchor_multipliers,
                    stage_terminal_weights=cfg.stage_terminal_weights,
                    stage_shrink_multipliers=cfg.stage_shrink_multipliers,
                )
            losses.append(float(loss.item()))
            maes.append(float(stats.get("mae_duration", np.nan)))
    return {
        "loss": float(np.nanmean(losses)) if losses else float("nan"),
        "mae_duration": float(np.nanmean(maes)) if maes else float("nan"),
    }


def _train_one_epoch(model_name: str, model: torch.nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> dict[str, float]:
    cfg = _learned_training_config(model_name)
    losses: list[float] = []
    maes: list[float] = []
    model.train()
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        inputs = {
            "weather_seq": batch["weather_seq"],
            "stage_index": batch["stage_index"],
            "base_dvr_seq": batch["base_dvr_seq"],
            "mask": batch["mask"],
        }
        if model_name == "m1_dvr_con":
            inputs["stage_state"] = batch["stage_state"]
        outputs = model(**inputs)
        if model_name == "m1_dvr_con":
            loss, stats = compute_m1_dvr_con_loss(
                outputs,
                batch["true_duration"],
                batch["mask"],
                model=model,
                stage_index=batch["stage_index"],
                event_loss_weight=cfg.event_loss_weight,
                terminal_loss_weight=cfg.terminal_loss_weight,
                shrink_loss_weight=cfg.shrink_loss_weight,
                smooth_loss_weight=cfg.smooth_loss_weight,
                mean_anchor_loss_weight=cfg.mean_anchor_loss_weight,
                stage_anchor_multipliers=cfg.stage_anchor_multipliers,
                stage_terminal_weights=cfg.stage_terminal_weights,
                stage_shrink_multipliers=cfg.stage_shrink_multipliers,
                gate_prior_weight=cfg.gate_prior_weight,
                gate_monotonic_weight=cfg.gate_monotonic_weight,
            )
        else:
            loss, stats = compute_dvr_loss(
                outputs,
                batch["true_duration"],
                batch["mask"],
                stage_index=batch["stage_index"],
                event_loss_weight=cfg.event_loss_weight,
                terminal_loss_weight=cfg.terminal_loss_weight,
                shrink_loss_weight=cfg.shrink_loss_weight,
                smooth_loss_weight=cfg.smooth_loss_weight,
                mean_anchor_loss_weight=cfg.mean_anchor_loss_weight,
                stage_anchor_multipliers=cfg.stage_anchor_multipliers,
                stage_terminal_weights=cfg.stage_terminal_weights,
                stage_shrink_multipliers=cfg.stage_shrink_multipliers,
            )
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        maes.append(float(stats.get("mae_duration", np.nan)))
    return {
        "loss": float(np.nanmean(losses)) if losses else float("nan"),
        "mae_duration": float(np.nanmean(maes)) if maes else float("nan"),
    }


def _make_stage_dataset(
    modeling_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    stage_requirements: dict[str, float] | None = None,
    stage_rate_priors: dict[str, float] | None = None,
) -> RiceDvrStageDataset:
    return RiceDvrStageDataset(
        modeling_df,
        weather_df,
        stage_requirements=stage_requirements,
        stage_rate_priors=stage_rate_priors,
        weather_features=DEFAULT_WEATHER_FEATURES,
    )


def _train_learned_model(
    model_name: str,
    train_core_df: pd.DataFrame,
    val_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    seed: int | None,
    device: str | torch.device | None,
) -> tuple[torch.nn.Module, dict[str, float], pd.DataFrame]:
    _set_random_seed(seed)
    cfg = _learned_training_config(model_name)
    torch_device = _resolve_torch_device(device)
    requirement_dataset = _make_stage_dataset(train_core_df, weather_df)
    stage_requirements = estimate_stage_requirements(requirement_dataset.samples)
    stage_rate_priors = estimate_stage_rate_priors(requirement_dataset.samples)
    if not stage_requirements:
        raise ValueError(f"No valid stage samples for {model_name}.")
    train_dataset = _make_stage_dataset(
        train_core_df,
        weather_df,
        stage_requirements=stage_requirements,
        stage_rate_priors=stage_rate_priors,
    )
    val_dataset = _make_stage_dataset(
        val_df if not val_df.empty else train_core_df,
        weather_df,
        stage_requirements=stage_requirements,
        stage_rate_priors=stage_rate_priors,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_dvr_batches,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_dvr_batches,
        num_workers=0,
    )
    model = _new_learned_model(model_name).to(torch_device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metric = float("inf")
    patience_left = int(cfg.early_stopping_patience)
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, int(cfg.epochs) + 1):
        train_stats = _train_one_epoch(model_name, model, train_loader, optimizer, torch_device)
        val_stats = _evaluate_loader(model_name, model, val_loader, torch_device)
        metric = float(val_stats["loss"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_mae_duration": train_stats["mae_duration"],
                "val_loss": val_stats["loss"],
                "val_mae_duration": val_stats["mae_duration"],
            }
        )
        if np.isfinite(metric) and metric < best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(cfg.early_stopping_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, stage_requirements, pd.DataFrame(history_rows)


def _prepare_rollout_weather(weather_df: pd.DataFrame, year: int, latitude: float) -> pd.DataFrame:
    season_weather = weather_df[weather_df["Date"].dt.year == int(year)].copy()
    if season_weather.empty:
        season_weather = weather_df.copy()
    if "daylength" not in season_weather.columns:
        from rice_phenology_hypernet.data.daylength import DayLengthCalculator

        calculator = DayLengthCalculator()
        season_weather["daylength"] = [
            calculator.day_length(date.year, date.month, date.day, float(latitude)) for date in season_weather["Date"]
        ]
    season_weather["doy"] = season_weather["Date"].dt.dayofyear.astype(int)
    return season_weather.sort_values("doy").reset_index(drop=True)


def _predict_one_season(
    row: pd.Series,
    weather_index: dict[tuple[int, int], pd.DataFrame],
    stage_requirements: dict[str, float],
    *,
    model_name: str,
    model: torch.nn.Module | None = None,
    max_sequence_length: int = 120,
) -> list[float]:
    key = (int(row["SID"]), int(row["year"]))
    if key not in weather_index:
        return [float("nan")] * len(PREDICTION_STAGE_NAMES)
    start_doy = _finite_or(float(row.get("obs_reviving", np.nan)), np.nan)
    if not np.isfinite(start_doy):
        return [float("nan")] * len(PREDICTION_STAGE_NAMES)
    season_weather = _prepare_rollout_weather(weather_index[key], int(row["year"]), float(row["latitude"]))
    predictions: list[float] = []
    transplanting_doy = _finite_or(float(row.get("transplanting_doy", start_doy)), start_doy)
    for stage_index, stage_name in enumerate(PREDICTION_STAGE_NAMES):
        stage_requirement = float(stage_requirements.get(stage_name, np.nan))
        if not np.isfinite(stage_requirement) or stage_requirement <= 0:
            predictions.append(float("nan"))
            start_doy += 1.0
            continue
        seq = season_weather[season_weather["doy"] >= int(round(start_doy))].head(int(max_sequence_length)).copy()
        if seq.empty:
            predictions.append(float("nan"))
            start_doy += 1.0
            continue
        if model_name == "m0_t":
            progress = np.cumsum(_temperature_dvr_sequence(seq, stage_requirement))
            duration = _predict_duration_from_progress(progress)
        elif model_name == "m0_dvr":
            progress = np.cumsum(_base_dvr_sequence(seq, stage_name, stage_requirement))
            duration = _predict_duration_from_progress(progress)
        else:
            assert model is not None
            device = _model_device(model)
            base = _base_dvr_sequence(seq, stage_name, stage_requirement)
            inputs: dict[str, torch.Tensor] = {
                "weather_seq": torch.tensor(seq.loc[:, DEFAULT_WEATHER_FEATURES].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(device),
                "stage_index": torch.tensor([stage_index], dtype=torch.long).to(device),
                "base_dvr_seq": torch.tensor(base, dtype=torch.float32).unsqueeze(0).to(device),
                "mask": torch.ones((1, len(seq)), dtype=torch.bool).to(device),
            }
            if model_name == "m1_dvr_con":
                inputs["stage_state"] = torch.tensor([[float(start_doy), float(start_doy - transplanting_doy)]], dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            duration = int(first_crossing_day(outputs["cum_progress_seq"], inputs["mask"]).detach().cpu().item())
        pred_doy = float(start_doy + duration - 1)
        predictions.append(pred_doy)
        start_doy = pred_doy + 1.0
    return predictions


def _make_prediction_rows(base_df: pd.DataFrame, preds: np.ndarray, task: str, fold: int, model: str, label: str) -> pd.DataFrame:
    out = base_df[["SID", "year"]].copy().reset_index(drop=True)
    out["task"] = task
    out["fold"] = fold
    out["model"] = model
    out["label"] = label
    for idx, stage in enumerate(PREDICTION_STAGE_NAMES):
        out[f"obs_{stage}"] = pd.to_numeric(base_df[f"obs_{stage}"], errors="coerce").to_numpy(dtype=float)
        out[f"pred_{stage}"] = preds[:, idx]
    return out


def _calculate_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return calculate_metrics_frame(predictions, group_cols=("task", "fold", "model"))


def _run_dvr_fold(
    *,
    task: str,
    model_name: str,
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    output_dir: Path,
    seed: int | None,
    device: str | torch.device | None,
) -> DvrExperimentBundle:
    weather_index = _build_weather_index(weather_df)
    if model_name == "m0_t":
        stage_requirements = estimate_stage_requirements_t(train_df, weather_df)
        model = None
        epoch_history = pd.DataFrame()
        split_summary = pd.DataFrame()
    elif model_name == "m0_dvr":
        stage_requirements = estimate_stage_requirements(_make_stage_dataset(train_df, weather_df).samples)
        model = None
        epoch_history = pd.DataFrame()
        split_summary = pd.DataFrame()
    else:
        cfg = _learned_training_config(model_name)
        train_core_df, val_df, split_meta = _build_inner_validation_split(
            train_df,
            task=task,
            outer_fold=fold,
            validation_fraction=cfg.validation_fraction,
            seed=seed,
        )
        model, stage_requirements, epoch_history = _train_learned_model(
            model_name,
            train_core_df,
            val_df,
            weather_df,
            seed=seed,
            device=device,
        )
        split_summary = pd.DataFrame([{**split_meta, "train_core_n": len(train_core_df), "validation_n": len(val_df)}])
        checkpoint_path = output_dir / f"{task}_{model_name}_fold{fold}_best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model_name,
                "model_config": model.config.__dict__,
                "stage_requirements": stage_requirements,
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            },
            checkpoint_path,
        )
    preds = np.asarray(
        [
            _predict_one_season(
                row,
                weather_index,
                stage_requirements,
                model_name=model_name,
                model=model,
                max_sequence_length=getattr(_learned_training_config(model_name), "max_sequence_length", 120)
                if model_name in {"m1_v2_dvr", "m1_dvr_con"}
                else 120,
            )
            for _, row in test_df.iterrows()
        ],
        dtype=float,
    )
    predictions = _make_prediction_rows(test_df, preds, task, fold, model_name, "test")
    metrics = _calculate_fold_metrics(predictions)
    diagnostics = {
        "epoch_history": epoch_history.assign(task=task, fold=fold, model=model_name) if not epoch_history.empty else epoch_history,
        "split_summary": split_summary.assign(task=task, fold=fold, model=model_name) if not split_summary.empty else split_summary,
        "stage_requirements": stage_requirements,
    }
    return DvrExperimentBundle(predictions=predictions, metrics=metrics, diagnostics=diagnostics)


def _finalize_dvr_experiment_outputs(
    task: str,
    model_name: str,
    bundles: list[DvrExperimentBundle],
    output_dir: Path,
    *,
    seed: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat([bundle.predictions for bundle in bundles], ignore_index=True) if bundles else pd.DataFrame()
    fold_metrics = pd.concat([bundle.metrics for bundle in bundles], ignore_index=True) if bundles else pd.DataFrame()
    metrics = calculate_metrics_frame(predictions, group_cols=("task", "model")) if not predictions.empty else pd.DataFrame()
    epoch_history = pd.concat(
        [bundle.diagnostics["epoch_history"] for bundle in bundles if not bundle.diagnostics["epoch_history"].empty],
        ignore_index=True,
    ) if bundles else pd.DataFrame()
    split_summary = pd.concat(
        [bundle.diagnostics["split_summary"] for bundle in bundles if not bundle.diagnostics["split_summary"].empty],
        ignore_index=True,
    ) if bundles else pd.DataFrame()
    predictions.to_csv(output_dir / f"{task}_{model_name}_predictions.csv", index=False)
    metrics.to_csv(output_dir / f"{task}_{model_name}_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / f"{task}_{model_name}_fold_metrics.csv", index=False)
    epoch_history.to_csv(output_dir / f"{task}_{model_name}_epoch_history.csv", index=False)
    split_summary.to_csv(output_dir / f"{task}_{model_name}_split_summary.csv", index=False)
    metadata = {
        "task": task,
        "model": model_name,
        "seed": seed,
        "folds": [
            {"fold": int(bundle.predictions["fold"].iloc[0]), "stage_requirements": bundle.diagnostics["stage_requirements"]}
            for bundle in bundles
            if not bundle.predictions.empty
        ],
    }
    (output_dir / f"{task}_{model_name}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _resolve_dvr_batch_seeds(seeds: tuple[int, ...] | list[int] | None) -> tuple[tuple[int, ...], str]:
    if seeds is None:
        return tuple(get_project_config().experiment.dvr_batch.seeds), "config"
    resolved = tuple(int(value) for value in seeds)
    if not 1 <= len(resolved) <= 10:
        raise ValueError("DVR batch seeds must contain between 1 and 10 integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError("DVR batch seeds must contain unique integers")
    return resolved, "cli"


def run_dvr_experiment(
    task: str,
    model_name: str,
    *,
    force: bool = False,
    run_id: str | None = None,
    seed: int | None = None,
    output_dir: Path | None = None,
    device: str | torch.device | None = "auto",
) -> None:
    del force
    if task not in MAIN_DVR_TASKS:
        raise ValueError(f"Unsupported DVR task={task}")
    if model_name not in PUBLIC_DVR_MODEL_NAMES:
        raise ValueError(f"Unsupported DVR model={model_name}")
    run_paths = initialize_run(run_id=run_id)
    target_dir = Path(output_dir) if output_dir is not None else run_paths.eval_dir
    register_experiment(run_paths.run_id, task, model_name, seed=seed, output_dir=target_dir)
    weather_df, _phenology_df = load_clean_data()
    modeling_df = build_modeling_dataset(force=False)
    splitter = _build_splitter(task)
    bundles: list[DvrExperimentBundle] = []
    for fold, train_idx, test_idx in splitter(modeling_df):
        train_df = modeling_df.iloc[train_idx].reset_index(drop=True)
        test_df = modeling_df.iloc[test_idx].reset_index(drop=True)
        bundles.append(
            _run_dvr_fold(
                task=task,
                model_name=model_name,
                fold=int(fold),
                train_df=train_df,
                test_df=test_df,
                weather_df=weather_df,
                output_dir=target_dir,
                seed=seed,
                device=device,
            )
        )
    _finalize_dvr_experiment_outputs(task, model_name, bundles, target_dir, seed=seed)


def run_all_dvr_experiments(
    *,
    force: bool = False,
    run_id: str | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    num_workers: int | None = None,
    threads_per_worker: int = 1,
    device: str | torch.device | None = "auto",
    gpu_workers: int = 1,
) -> None:
    del force, num_workers, gpu_workers
    resolved_seeds, seed_source = _resolve_dvr_batch_seeds(seeds)
    run_paths = initialize_run(run_id=run_id)
    weather_df, _phenology_df = load_clean_data()
    modeling_df = build_modeling_dataset(force=False)
    with _thread_environment_override(threads_per_worker):
        for seed in resolved_seeds:
            seed_dir = run_paths.eval_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for task in MAIN_DVR_TASKS:
                audit = collect_protocol_audit(modeling_df, task)
                audit_dir = run_paths.eval_dir / "evaluation_protocol_audit"
                audit_dir.mkdir(parents=True, exist_ok=True)
                for name, frame in audit.items():
                    frame.to_csv(audit_dir / f"{task}_{name}.csv", index=False)
                splitter = _build_splitter(task)
                for model_name in PUBLIC_DVR_MODEL_NAMES:
                    bundles = []
                    for fold, train_idx, test_idx in splitter(modeling_df):
                        bundles.append(
                            _run_dvr_fold(
                                task=task,
                                model_name=model_name,
                                fold=int(fold),
                                train_df=modeling_df.iloc[train_idx].reset_index(drop=True),
                                test_df=modeling_df.iloc[test_idx].reset_index(drop=True),
                                weather_df=weather_df,
                                output_dir=seed_dir,
                                seed=int(seed),
                                device=device,
                            )
                        )
                    _finalize_dvr_experiment_outputs(task, model_name, bundles, seed_dir, seed=int(seed))
                    register_experiment(run_paths.run_id, task, model_name, seed=int(seed), output_dir=seed_dir)
            build_dvr_relative_change_summary(seed_dir, tasks=MAIN_DVR_TASKS, models=PUBLIC_DVR_MODEL_NAMES)
    aggregate_dvr_relative_change_summaries(run_paths.eval_dir, resolved_seeds, tasks=MAIN_DVR_TASKS)
    update_run_metadata(
        run_paths.run_id,
        dvr_batch={"seeds": list(resolved_seeds), "seed_source": seed_source, "models": list(PUBLIC_DVR_MODEL_NAMES), "tasks": list(MAIN_DVR_TASKS)},
    )


def _deployment_seed_dir(run_id: str, seed: int) -> Path:
    return SETTINGS.models_dir / run_id / f"seed_{int(seed)}"


def _deployment_model_dir(run_id: str, model_name: str, seed: int) -> Path:
    return _deployment_seed_dir(run_id, seed) / model_name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_learned_dvr_deployment_model(model_name: str, modeling_df: pd.DataFrame, weather_df: pd.DataFrame, *, seed: int) -> tuple[DvrDeploymentArtifact, pd.DataFrame]:
    cfg = _learned_training_config(model_name)
    train_core, val_df, _meta = _build_deployment_inner_validation_split(
        modeling_df,
        validation_fraction=cfg.validation_fraction,
        seed=seed,
    )
    model, stage_requirements, history = _train_learned_model(
        model_name,
        train_core,
        val_df,
        weather_df,
        seed=seed,
        device="auto",
    )
    background_gates = None
    if isinstance(model, M1ConDvrModel):
        background_gates = [float(value) for value in model.get_background_gates().detach().cpu().numpy()]
    artifact = DvrDeploymentArtifact(
        artifact_type="learned",
        model_name=model_name,
        stage_requirements={stage: float(value) for stage, value in stage_requirements.items()},
        model_config=model.config.__dict__,
        model_state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        background_gate_values=background_gates,
        metadata={"seed": int(seed), "best_epoch": int(history.iloc[history["val_loss"].idxmin()]["epoch"]) if not history.empty else None},
    )
    return artifact, history


def _save_deployment_artifact(run_id: str, artifact: DvrDeploymentArtifact, seed: int, epoch_history: pd.DataFrame | None = None) -> None:
    model_dir = _deployment_model_dir(run_id, artifact.model_name, seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(model_dir / "stage_requirements.json", {stage: float(value) for stage, value in artifact.stage_requirements.items()})
    metadata = dict(artifact.metadata or {})
    metadata.update({"artifact_type": artifact.artifact_type, "model_name": artifact.model_name})
    if artifact.background_gate_values is not None:
        metadata["background_gate_values"] = artifact.background_gate_values
    _write_json(model_dir / "deployment_metadata.json", metadata)
    if artifact.artifact_type == "learned":
        torch.save(
            {
                "model": artifact.model_name,
                "model_config": artifact.model_config,
                "model_state_dict": artifact.model_state_dict,
                "stage_requirements": artifact.stage_requirements,
                "background_gate_values": artifact.background_gate_values,
            },
            model_dir / "model.pt",
        )
        if epoch_history is not None:
            epoch_history.to_csv(model_dir / "deployment_epoch_history.csv", index=False)


def train_dvr_deployment_models(*, run_id: str | None = None, seed: int = 61) -> dict[str, DvrDeploymentArtifact]:
    effective_run_id = run_id or "paper_deployment"
    weather_df, _phenology_df = load_clean_data()
    modeling_df = build_modeling_dataset(force=False)
    process_artifacts = {
        "m0_t": DvrDeploymentArtifact(
            artifact_type="process",
            model_name="m0_t",
            stage_requirements=estimate_stage_requirements_t(modeling_df, weather_df),
            metadata={"seed": int(seed)},
        ),
        "m0_dvr": DvrDeploymentArtifact(
            artifact_type="process",
            model_name="m0_dvr",
            stage_requirements=estimate_stage_requirements(_make_stage_dataset(modeling_df, weather_df).samples),
            metadata={"seed": int(seed)},
        ),
    }
    results: dict[str, DvrDeploymentArtifact] = {}
    for artifact in process_artifacts.values():
        _save_deployment_artifact(effective_run_id, artifact, seed)
        results[artifact.model_name] = artifact
    for model_name in ("m1_v2_dvr", "m1_dvr_con"):
        artifact, history = _train_learned_dvr_deployment_model(model_name, modeling_df, weather_df, seed=seed)
        _save_deployment_artifact(effective_run_id, artifact, seed, epoch_history=history)
        results[model_name] = artifact
    manifest = {
        "run_id": effective_run_id,
        "seed": int(seed),
        "models": list(DEPLOYMENT_MODEL_NAMES),
    }
    _write_json(SETTINGS.models_dir / effective_run_id / "deployment_manifest.json", manifest)
    return results


def load_dvr_deployment_artifact(run_id: str, model_name: str, *, seed: int) -> DvrDeploymentArtifact:
    if model_name not in DEPLOYMENT_MODEL_NAMES:
        raise ValueError(f"Unsupported deployment model={model_name}")
    artifact_dir = _deployment_model_dir(run_id, model_name, seed)
    requirements_path = artifact_dir / "stage_requirements.json"
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing deployment stage requirements: {requirements_path}")
    stage_requirements = {stage: float(value) for stage, value in json.loads(requirements_path.read_text(encoding="utf-8")).items()}
    metadata_path = artifact_dir / "deployment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if model_name in {"m0_t", "m0_dvr"}:
        return DvrDeploymentArtifact(
            artifact_type="process",
            model_name=model_name,
            stage_requirements=stage_requirements,
            artifact_dir=artifact_dir,
            metadata=metadata,
        )
    checkpoint_path = artifact_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing deployment model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return DvrDeploymentArtifact(
        artifact_type="learned",
        model_name=model_name,
        stage_requirements=stage_requirements,
        model_config=dict(checkpoint.get("model_config") or {}),
        model_state_dict=dict(checkpoint.get("model_state_dict") or {}),
        background_gate_values=checkpoint.get("background_gate_values"),
        artifact_dir=artifact_dir,
        metadata=metadata,
    )


def materialize_dvr_deployment_artifact(artifact: DvrDeploymentArtifact) -> MaterializedProcessModel | torch.nn.Module:
    if artifact.artifact_type == "process":
        return MaterializedProcessModel(model_name=artifact.model_name, stage_requirements=artifact.stage_requirements)
    if artifact.artifact_type != "learned":
        raise ValueError(f"Unsupported deployment artifact_type={artifact.artifact_type}")
    if artifact.model_name == "m1_v2_dvr":
        config = M1V2DvrConfig(**(artifact.model_config or {}))
        model = M1V2DvrModel(config=config)
    elif artifact.model_name == "m1_dvr_con":
        config = M1ConDvrConfig(**(artifact.model_config or {}))
        model = M1ConDvrModel(config=config)
    else:
        raise ValueError(f"Unsupported learned deployment model={artifact.model_name}")
    if not artifact.model_state_dict:
        raise ValueError(f"Learned artifact missing model_state_dict: {artifact.model_name}")
    model.load_state_dict(artifact.model_state_dict)
    model.eval()
    return model


def _load_task_checkpoints(eval_dir: Path, task: str, model_name: str, seed: int | None = None) -> list[dict[str, Any]]:
    del seed
    payloads: list[dict[str, Any]] = []
    for path in sorted(Path(eval_dir).glob(f"{task}_{model_name}_fold*_best.pt")):
        payloads.append(torch.load(path, map_location="cpu"))
    return payloads
