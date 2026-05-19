from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rice_phenology_hypernet.data.dataset_dvr import DEFAULT_WEATHER_FEATURES, DVR_STAGE_NAMES, RiceDvrStageDataset
from rice_phenology_hypernet.data.io import load_clean_data
from rice_phenology_hypernet.features import build_modeling_dataset
from rice_phenology_hypernet.runtime import RunPaths, initialize_run

from .runner_dvr import _base_dvr_sequence, load_dvr_deployment_artifact, materialize_dvr_deployment_artifact


MODEL_NAME = "m1_dvr_con"
DEFAULT_DEPLOYMENT_RUN_ID = "molde4_seed61"
DEFAULT_SEED = 61
DEFAULT_STAGE = "heading"
DEFAULT_ANALYSIS_STAGES = DVR_STAGE_NAMES
DEFAULT_FIGURE_STAGES = ("booting", "heading", "maturity")
DEFAULT_N_BOOT = 1000
DEFAULT_RANDOM_SEED = 20260328
TEMPERATURE_OFFSETS = (-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0)
DAYLENGTH_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PRECIPITATION_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)
PERTURBATION_SPECS = (
    ("temperature", "temperature_offset_c", "deg C", TEMPERATURE_OFFSETS),
    ("daylength", "daylength_offset_h", "h", DAYLENGTH_OFFSETS),
    ("precipitation", "precipitation_multiplier", "x", PRECIPITATION_MULTIPLIERS),
)


@dataclass(frozen=True)
class ModifierInterpretabilityResult:
    run_id: str
    stages: tuple[str, ...]
    sample_paths: dict[str, Path]
    summary_paths: dict[str, Path]
    metadata_paths: dict[str, Path]
    figure_path: Path | None
    sample_frame: pd.DataFrame
    summary_frame: pd.DataFrame

    @property
    def sample_path(self) -> Path:
        return self.sample_paths[self.stages[0]]

    @property
    def summary_path(self) -> Path:
        return self.summary_paths[self.stages[0]]

    @property
    def metadata_path(self) -> Path:
        return self.metadata_paths[self.stages[0]]


def modifier_interpretability_dir(eval_dir: Path) -> Path:
    return Path(eval_dir) / "modifier_interpretability"


def modifier_interpretability_prefix(stage: str) -> str:
    return f"pghm_con_{stage}_modifier_perturbation"


def modifier_interpretability_summary_path(eval_dir: Path, *, stage: str = DEFAULT_STAGE) -> Path:
    return modifier_interpretability_dir(eval_dir) / f"{modifier_interpretability_prefix(stage)}_summary.csv"


def modifier_interpretability_sample_path(eval_dir: Path, *, stage: str = DEFAULT_STAGE) -> Path:
    return modifier_interpretability_dir(eval_dir) / f"{modifier_interpretability_prefix(stage)}_samples.csv"


def _stage_transition_label(stage: str) -> str:
    if stage not in DVR_STAGE_NAMES:
        raise ValueError(f"Unsupported stage '{stage}'. Expected one of: {', '.join(DVR_STAGE_NAMES)}")
    index = DVR_STAGE_NAMES.index(stage)
    if index == 0:
        return f"reviving_to_{stage}"
    return f"{DVR_STAGE_NAMES[index - 1]}_to_{stage}"


def _sample_id(sample: dict[str, object]) -> str:
    return "_".join(
        [
            str(int(sample["sid"])),
            str(int(sample["year"])),
            str(sample["stage_name"]),
            str(int(sample["start_doy"])),
            str(int(sample["end_doy"])),
        ]
    )


def _weather_frame_from_sample(sample: dict[str, object]) -> pd.DataFrame:
    values = sample["weather_seq"].detach().cpu().numpy()
    return pd.DataFrame(values, columns=DEFAULT_WEATHER_FEATURES)


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device)


def _resolve_analysis_stages(stage: str | None) -> tuple[str, ...]:
    if stage is None:
        return tuple(DEFAULT_ANALYSIS_STAGES)
    if stage not in DVR_STAGE_NAMES:
        raise ValueError(f"Unsupported stage '{stage}'. Expected one of: {', '.join(DVR_STAGE_NAMES)}")
    return (stage,)


def _log_modifier_sequence(
    *,
    model: Any,
    weather: pd.DataFrame,
    base_dvr: np.ndarray,
    sample: dict[str, object],
) -> np.ndarray:
    device = _model_device(model)
    batch = {
        "weather_seq": _move_tensor(
            torch.tensor(weather.loc[:, DEFAULT_WEATHER_FEATURES].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0),
            device,
        ),
        "stage_state": _move_tensor(sample["stage_state"].detach().clone().unsqueeze(0), device),
        "stage_index": _move_tensor(torch.tensor([int(sample["stage_index"])], dtype=torch.long), device),
        "base_dvr_seq": _move_tensor(torch.tensor(base_dvr, dtype=torch.float32).unsqueeze(0), device),
        "mask": _move_tensor(torch.ones((1, len(weather)), dtype=torch.bool), device),
    }
    with torch.no_grad():
        outputs = model(**batch)
    log_modifier = outputs["log_modifier_seq"].squeeze(0).detach().cpu().numpy()
    return np.asarray(log_modifier, dtype=float)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _dvr_star_sequence(base_dvr: np.ndarray, log_modifier: np.ndarray) -> np.ndarray:
    if len(base_dvr) != len(log_modifier):
        raise ValueError(f"base DVR and log modifier length mismatch: {len(base_dvr)} != {len(log_modifier)}")
    return np.asarray(base_dvr, dtype=float) * np.exp(np.asarray(log_modifier, dtype=float))


def _apply_perturbation(weather: pd.DataFrame, input_group: str, value: float) -> pd.DataFrame:
    perturbed = weather.copy()
    if input_group == "temperature":
        for column in ("TemAver", "TemMin", "TemMax"):
            perturbed[column] = pd.to_numeric(perturbed[column], errors="coerce") + float(value)
        return perturbed
    if input_group == "daylength":
        perturbed["daylength"] = np.clip(pd.to_numeric(perturbed["daylength"], errors="coerce") + float(value), 0.0, 24.0)
        return perturbed
    if input_group == "precipitation":
        perturbed["Precipitation"] = np.maximum(0.0, pd.to_numeric(perturbed["Precipitation"], errors="coerce") * float(value))
        return perturbed
    raise ValueError(f"Unsupported perturbation group: {input_group}")


def _evaluate_sample_perturbations(
    sample: dict[str, object],
    *,
    model: Any,
    stage_requirement: float,
    stage_label: str,
) -> list[dict[str, object]]:
    stage_name = str(sample["stage_name"])
    weather = _weather_frame_from_sample(sample)
    original_base_dvr = _base_dvr_sequence(weather, stage_name, stage_requirement)
    original_log_modifier = _log_modifier_sequence(
        model=model,
        weather=weather,
        base_dvr=original_base_dvr,
        sample=sample,
    )
    original_mean_log_modifier = _mean(original_log_modifier)
    original_dvr_star = _dvr_star_sequence(original_base_dvr, original_log_modifier)
    rows: list[dict[str, object]] = []
    for input_group, value_column, unit, values in PERTURBATION_SPECS:
        for value in values:
            perturbed_weather = _apply_perturbation(weather, input_group, float(value))
            perturbed_base_dvr = _base_dvr_sequence(perturbed_weather, stage_name, stage_requirement)
            perturbed_log_modifier = _log_modifier_sequence(
                model=model,
                weather=perturbed_weather,
                base_dvr=perturbed_base_dvr,
                sample=sample,
            )
            perturbed_mean_log_modifier = _mean(perturbed_log_modifier)
            perturbed_dvr_star = _dvr_star_sequence(perturbed_base_dvr, perturbed_log_modifier)
            rows.append(
                {
                    "sample_id": _sample_id(sample),
                    "SID": int(sample["sid"]),
                    "year": int(sample["year"]),
                    "stage": stage_name,
                    "stage_label": stage_label,
                    "start_doy": int(sample["start_doy"]),
                    "end_doy": int(sample["end_doy"]),
                    "n_days": int(sample["true_duration"]),
                    "input_group": input_group,
                    "value_column": value_column,
                    "perturbation_value": float(value),
                    "perturbation_unit": unit,
                    "original_mean_log_modifier": original_mean_log_modifier,
                    "perturbed_mean_log_modifier": perturbed_mean_log_modifier,
                    "delta_log_modifier": float(perturbed_mean_log_modifier - original_mean_log_modifier),
                    "original_base_dvr_mean": float(np.mean(original_base_dvr)),
                    "perturbed_base_dvr_mean": float(np.mean(perturbed_base_dvr)),
                    "original_base_dvr_sum": float(np.sum(original_base_dvr)),
                    "perturbed_base_dvr_sum": float(np.sum(perturbed_base_dvr)),
                    "original_dvr_star_mean": _mean(original_dvr_star),
                    "perturbed_dvr_star_mean": _mean(perturbed_dvr_star),
                    "delta_dvr_star_mean": float(_mean(perturbed_dvr_star) - _mean(original_dvr_star)),
                    "original_dvr_star_sum": float(np.sum(original_dvr_star)),
                    "perturbed_dvr_star_sum": float(np.sum(perturbed_dvr_star)),
                    "delta_dvr_star_sum": float(np.sum(perturbed_dvr_star) - np.sum(original_dvr_star)),
                }
            )
    return rows


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1 or n_boot <= 0:
        value = float(clean.mean())
        return value, value
    indices = rng.integers(0, clean.size, size=(int(n_boot), clean.size))
    means = clean[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize_modifier_perturbations(
    sample_frame: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []
    sort_cols = ["input_group", "perturbation_value"]
    for keys, group in sample_frame.sort_values(sort_cols).groupby(sort_cols, sort=False):
        input_group, perturbation_value = keys
        values = pd.to_numeric(group["delta_log_modifier"], errors="coerce").to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(values, rng=rng, n_boot=n_boot)
        rows.append(
            {
                "input_group": input_group,
                "value_column": str(group["value_column"].iloc[0]),
                "perturbation_value": float(perturbation_value),
                "perturbation_unit": str(group["perturbation_unit"].iloc[0]),
                "stage": str(group["stage"].iloc[0]),
                "stage_label": str(group["stage_label"].iloc[0]),
                "delta_log_modifier_mean": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
                "delta_log_modifier_std": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0,
                "delta_log_modifier_ci_low": ci_low,
                "delta_log_modifier_ci_high": ci_high,
                "n": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_modifier_interpretability(
    *,
    deployment_run_id: str = DEFAULT_DEPLOYMENT_RUN_ID,
    run_id: str | None = None,
    seed: int = DEFAULT_SEED,
    stage: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    build_figure: bool = True,
) -> ModifierInterpretabilityResult:
    effective_run_id = run_id or deployment_run_id
    run_paths = initialize_run(run_id=effective_run_id)
    output_dir = modifier_interpretability_dir(run_paths.eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_dvr_deployment_artifact(deployment_run_id, MODEL_NAME, seed=seed)
    model = materialize_dvr_deployment_artifact(artifact)
    model.eval()

    weather_df, _phenology_df = load_clean_data()
    modeling_df = build_modeling_dataset(force=False)
    dataset = RiceDvrStageDataset(modeling_df, weather_df, stage_requirements=artifact.stage_requirements)
    stages = _resolve_analysis_stages(stage)

    sample_paths: dict[str, Path] = {}
    summary_paths: dict[str, Path] = {}
    metadata_paths: dict[str, Path] = {}
    sample_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for stage_name in stages:
        stage_label = _stage_transition_label(stage_name)
        selected_samples = [sample for sample in dataset.samples if str(sample["stage_name"]) == stage_name]
        if not selected_samples:
            raise ValueError(f"No historical DVR stage samples found for stage '{stage_name}'.")

        rows: list[dict[str, object]] = []
        stage_requirement = float(artifact.stage_requirements[stage_name])
        for sample in selected_samples:
            rows.extend(
                _evaluate_sample_perturbations(
                    sample,
                    model=model,
                    stage_requirement=stage_requirement,
                    stage_label=stage_label,
                )
            )

        sample_frame = pd.DataFrame(rows)
        summary_frame = _summarize_modifier_perturbations(sample_frame, n_boot=n_boot, random_seed=random_seed)
        prefix = modifier_interpretability_prefix(stage_name)
        sample_path = output_dir / f"{prefix}_samples.csv"
        summary_path = output_dir / f"{prefix}_summary.csv"
        metadata_path = output_dir / f"{prefix}_metadata.json"
        sample_frame.to_csv(sample_path, index=False)
        summary_frame.to_csv(summary_path, index=False)
        _write_metadata(
            metadata_path,
            {
                "deployment_run_id": deployment_run_id,
                "run_id": run_paths.run_id,
                "model": MODEL_NAME,
                "seed": int(seed),
                "stage": stage_name,
                "stage_label": stage_label,
                "analyzed_stages": list(stages),
                "n_stage_samples": int(len(selected_samples)),
                "n_boot": int(n_boot),
                "random_seed": int(random_seed),
                "temperature_offsets_c": list(TEMPERATURE_OFFSETS),
                "daylength_offsets_h": list(DAYLENGTH_OFFSETS),
                "precipitation_multipliers": list(PRECIPITATION_MULTIPLIERS),
                "interpretation_boundary": (
                    "Historical real-weather posterior perturbation from the deployed PGHM-CON model; "
                    "curves describe effective corrections to the photothermal DVR baseline, not physiological response functions."
                ),
            },
        )

        sample_paths[stage_name] = sample_path
        summary_paths[stage_name] = summary_path
        metadata_paths[stage_name] = metadata_path
        sample_frames.append(sample_frame)
        summary_frames.append(summary_frame)

    figure_path: Path | None = None
    if build_figure:
        from rice_phenology_hypernet.figures.builder import modifier_interpretability

        figure_stages = DEFAULT_FIGURE_STAGES
        if not all(modifier_interpretability_sample_path(run_paths.eval_dir, stage=item).exists() for item in figure_stages):
            figure_stages = stages
        figure_path = modifier_interpretability(run_paths.figures_dir, run_paths.eval_dir, stages=figure_stages)

    return ModifierInterpretabilityResult(
        run_id=run_paths.run_id,
        stages=stages,
        sample_paths=sample_paths,
        summary_paths=summary_paths,
        metadata_paths=metadata_paths,
        figure_path=figure_path,
        sample_frame=pd.concat(sample_frames, ignore_index=True),
        summary_frame=pd.concat(summary_frames, ignore_index=True),
    )


__all__ = [
    "ModifierInterpretabilityResult",
    "analyze_modifier_interpretability",
    "modifier_interpretability_dir",
    "modifier_interpretability_prefix",
    "modifier_interpretability_sample_path",
    "modifier_interpretability_summary_path",
]
