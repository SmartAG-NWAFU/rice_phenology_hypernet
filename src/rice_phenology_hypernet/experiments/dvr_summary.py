from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TASK_ORDER = ["sample", "site", "year"]
STAGE_ORDER = ["tillering", "jointing", "booting", "heading", "maturity", "all_stage"]
MODEL_ORDER = ["m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con"]
BASELINE_MODEL = "m0_dvr"
COMPARISON_MODELS = [model_name for model_name in MODEL_ORDER if model_name != BASELINE_MODEL]
SUMMARY_METRICS = ("mae", "rmse", "bias", "r2")
DERIVED_METRICS = (
    "mae_improve_days",
    "mae_improve_pct",
    "bias_shift",
    "abs_bias_improve_days",
    "abs_bias_improve_pct",
    "r2_gain",
)

SUMMARY_COLUMNS = [
    "task",
    "stage",
    *[f"{model_name}_{metric_name}" for model_name in MODEL_ORDER for metric_name in SUMMARY_METRICS],
    *[
        f"{model_name}_vs_{BASELINE_MODEL}_{metric_name}"
        for model_name in COMPARISON_MODELS
        for metric_name in DERIVED_METRICS
    ],
]


def _resolve_tasks(tasks: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if tasks is None:
        return list(TASK_ORDER)
    requested = [str(task) for task in tasks]
    invalid = sorted({task for task in requested if task not in TASK_ORDER})
    if invalid:
        raise ValueError(f"Unsupported DVR tasks: {', '.join(invalid)}")
    return [task for task in TASK_ORDER if task in requested]


def _resolve_models(models: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if models is None:
        return list(MODEL_ORDER)
    requested = [str(model_name) for model_name in models]
    invalid = sorted({model_name for model_name in requested if model_name not in MODEL_ORDER})
    if invalid:
        raise ValueError(f"Unsupported DVR models: {', '.join(invalid)}")
    return [model_name for model_name in MODEL_ORDER if model_name in requested]


def _required_metrics_path(eval_dir: Path, task: str, model_name: str) -> Path:
    path = eval_dir / f"{task}_{model_name}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required DVR metrics file: {path.name}")
    return path


def _normalize_metrics_frame(path: Path, task: str, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"stage", "mae", "rmse", "bias", "r2"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing_columns)}")
    metrics = frame.loc[frame["stage"].isin(STAGE_ORDER), ["stage", "mae", "rmse", "bias", "r2"]].copy()
    observed_stages = set(metrics["stage"].astype(str))
    missing_stages = [stage for stage in STAGE_ORDER if stage not in observed_stages]
    if missing_stages:
        raise ValueError(f"{path.name} is missing required stages: {', '.join(missing_stages)}")
    metrics["task"] = task
    metrics["model"] = model_name
    metrics["stage"] = pd.Categorical(metrics["stage"], categories=STAGE_ORDER, ordered=True)
    for column in SUMMARY_METRICS:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    return metrics.sort_values("stage").set_index("stage")


def _empty_metrics_frame(task: str, model_name: str) -> pd.DataFrame:
    metrics = pd.DataFrame({"stage": STAGE_ORDER, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan})
    metrics["task"] = task
    metrics["model"] = model_name
    metrics["stage"] = pd.Categorical(metrics["stage"], categories=STAGE_ORDER, ordered=True)
    return metrics.sort_values("stage").set_index("stage")


def _safe_pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(float(denominator)) <= 1e-12:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def _build_relative_row(baseline_metrics: pd.Series, model_metrics: pd.Series, model_name: str) -> dict[str, float]:
    baseline_values = pd.to_numeric(baseline_metrics.loc[list(SUMMARY_METRICS)], errors="coerce")
    model_values = pd.to_numeric(model_metrics.loc[list(SUMMARY_METRICS)], errors="coerce")
    if baseline_values.isna().any() or model_values.isna().any():
        return {f"{model_name}_vs_{BASELINE_MODEL}_{metric_name}": float("nan") for metric_name in DERIVED_METRICS}
    mae_improve_days = float(baseline_values["mae"] - model_values["mae"])
    abs_bias_improve_days = float(abs(baseline_values["bias"]) - abs(model_values["bias"]))
    return {
        f"{model_name}_vs_{BASELINE_MODEL}_mae_improve_days": mae_improve_days,
        f"{model_name}_vs_{BASELINE_MODEL}_mae_improve_pct": _safe_pct(mae_improve_days, float(baseline_values["mae"])),
        f"{model_name}_vs_{BASELINE_MODEL}_bias_shift": float(model_values["bias"] - baseline_values["bias"]),
        f"{model_name}_vs_{BASELINE_MODEL}_abs_bias_improve_days": abs_bias_improve_days,
        f"{model_name}_vs_{BASELINE_MODEL}_abs_bias_improve_pct": _safe_pct(abs_bias_improve_days, abs(float(baseline_values["bias"]))),
        f"{model_name}_vs_{BASELINE_MODEL}_r2_gain": float(model_values["r2"] - baseline_values["r2"]),
    }


def build_dvr_relative_change_summary(
    eval_dir: Path,
    tasks: list[str] | tuple[str, ...] | None = None,
    models: list[str] | tuple[str, ...] | None = None,
) -> Path:
    eval_dir = Path(eval_dir)
    resolved_tasks = _resolve_tasks(tasks)
    resolved_models = _resolve_models(models)
    rows: list[dict[str, float | str]] = []
    for task in resolved_tasks:
        metric_frames = {
            model_name: _normalize_metrics_frame(_required_metrics_path(eval_dir, task, model_name), task, model_name)
            for model_name in resolved_models
        }
        for model_name in MODEL_ORDER:
            metric_frames.setdefault(model_name, _empty_metrics_frame(task, model_name))
        for stage in STAGE_ORDER:
            baseline_metrics = metric_frames[BASELINE_MODEL].loc[stage]
            row: dict[str, float | str] = {"task": task, "stage": stage}
            for model_name in MODEL_ORDER:
                model_metrics = metric_frames[model_name].loc[stage]
                for metric_name in SUMMARY_METRICS:
                    row[f"{model_name}_{metric_name}"] = float(model_metrics[metric_name])
            for model_name in COMPARISON_MODELS:
                row.update(_build_relative_row(baseline_metrics, metric_frames[model_name].loc[stage], model_name))
            rows.append(row)
    output_path = eval_dir / "dvr_relative_change_summary.csv"
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(output_path, index=False)
    return output_path


def aggregate_dvr_relative_change_summaries(
    eval_dir: Path,
    seeds: tuple[int, ...] | list[int],
    tasks: list[str] | tuple[str, ...] | None = None,
) -> Path:
    eval_dir = Path(eval_dir)
    resolved_tasks = _resolve_tasks(tasks)
    frames = []
    for seed in tuple(int(value) for value in seeds):
        summary_path = eval_dir / f"seed_{seed}" / "dvr_relative_change_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing per-seed DVR summary for seed_{seed}: {summary_path}")
        frame = pd.read_csv(summary_path)
        frame["seed"] = int(seed)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    numeric_columns = [column for column in SUMMARY_COLUMNS if column not in {"task", "stage"}]
    rows: list[dict[str, float | str]] = []
    for task in resolved_tasks:
        for stage in STAGE_ORDER:
            subset = combined[(combined["task"] == task) & (combined["stage"] == stage)]
            row: dict[str, float | str] = {"task": task, "stage": stage}
            for column in numeric_columns:
                values = pd.to_numeric(subset[column], errors="coerce")
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
            rows.append(row)
    output_path = eval_dir / "dvr_relative_change_summary_aggregated.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
