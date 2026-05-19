from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rice_phenology_hypernet.evaluation import calculate_metrics_frame
from rice_phenology_hypernet.runtime import require_run


PAPER_TASKS = ("sample", "site", "year")
PAPER_MODELS = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")
STAGES = ("tillering", "jointing", "booting", "heading", "maturity")


@dataclass(frozen=True)
class DvrDiagnosticResult:
    task: str
    output_dir: Path
    output_paths: dict[str, Path]


def _prediction_path(eval_dir: Path, task: str, model_name: str) -> Path:
    return eval_dir / f"{task}_{model_name}_predictions.csv"


def _load_prediction_frames(eval_dir: Path, task: str) -> pd.DataFrame:
    frames = []
    missing = []
    for model_name in PAPER_MODELS:
        path = _prediction_path(eval_dir, task, model_name)
        if not path.exists():
            missing.append(path.name)
            continue
        frame = pd.read_csv(path)
        frame["task"] = task
        frame["model"] = model_name
        frames.append(frame)
    if missing:
        raise FileNotFoundError(f"Missing DVR prediction files for {task}: {', '.join(missing)}")
    return pd.concat(frames, ignore_index=True)


def _rollout_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    metrics = calculate_metrics_frame(predictions, group_cols=("task", "model"))
    metrics.insert(2, "mode", "rollout")
    return metrics


def _error_decomposition(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model_name), group in predictions.groupby(["task", "model"]):
        for stage in STAGES:
            obs = pd.to_numeric(group[f"obs_{stage}"], errors="coerce")
            pred = pd.to_numeric(group[f"pred_{stage}"], errors="coerce")
            err = pred - obs
            err = err[np.isfinite(err)]
            rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "stage": stage,
                    "mean_error": float(err.mean()) if len(err) else np.nan,
                    "median_error": float(err.median()) if len(err) else np.nan,
                    "mean_abs_error": float(err.abs().mean()) if len(err) else np.nan,
                    "n": int(len(err)),
                }
            )
    return pd.DataFrame(rows)


def _progress_at_truth_end(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in predictions.iterrows():
        for stage in STAGES:
            obs = pd.to_numeric(pd.Series([row.get(f"obs_{stage}")]), errors="coerce").iloc[0]
            pred = pd.to_numeric(pd.Series([row.get(f"pred_{stage}")]), errors="coerce").iloc[0]
            if not np.isfinite(obs) or not np.isfinite(pred):
                continue
            rows.append(
                {
                    "task": row["task"],
                    "model": row["model"],
                    "fold": row.get("fold", np.nan),
                    "SID": row.get("SID", np.nan),
                    "year": row.get("year", np.nan),
                    "stage": stage,
                    "obs_doy": float(obs),
                    "pred_doy": float(pred),
                    "progress_proxy_at_truth_end": float(obs - pred),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame(columns=["task", "model", "stage", "mean_progress_proxy_at_truth_end", "n"])
    else:
        summary = (
            detail.groupby(["task", "model", "stage"], as_index=False)
            .agg(
                mean_progress_proxy_at_truth_end=("progress_proxy_at_truth_end", "mean"),
                median_progress_proxy_at_truth_end=("progress_proxy_at_truth_end", "median"),
                n=("progress_proxy_at_truth_end", "size"),
            )
        )
    return detail, summary


def _empty_modifier_daily(task: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "task",
            "model",
            "stage",
            "fold",
            "SID",
            "year",
            "day_index",
            "base_dvr",
            "modifier",
            "modified_dvr",
        ]
    ).assign(task=task).iloc[0:0]


def _empty_modifier_summary(task: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["task", "model", "stage", "mean_modifier", "mean_modified_dvr", "n"]).assign(task=task).iloc[0:0]


def _early_stopping_audit(eval_dir: Path, task: str) -> pd.DataFrame:
    rows = []
    for model_name in ("m1_v2_dvr", "m1_dvr_con"):
        path = eval_dir / f"{task}_{model_name}_epoch_history.csv"
        if not path.exists():
            continue
        history = pd.read_csv(path)
        if history.empty:
            continue
        metric_column = "val_rollout_all_stage_mae" if "val_rollout_all_stage_mae" in history.columns else "val_loss"
        for fold, group in history.groupby("fold"):
            values = pd.to_numeric(group[metric_column], errors="coerce")
            best_idx = values.idxmin()
            rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "fold": int(fold),
                    "selection_metric": metric_column,
                    "best_epoch": int(history.loc[best_idx, "epoch"]),
                    "best_value": float(values.loc[best_idx]),
                    "epochs": int(len(group)),
                }
            )
    return pd.DataFrame(rows, columns=["task", "model", "fold", "selection_metric", "best_epoch", "best_value", "epochs"])


def _requirement_shift_audit(eval_dir: Path, task: str) -> pd.DataFrame:
    rows = []
    for model_name in PAPER_MODELS:
        path = eval_dir / f"{task}_{model_name}_metadata.json"
        if not path.exists():
            continue
        metadata = pd.read_json(path, typ="series")
        fold_requirements = metadata.get("fold_stage_requirements", [])
        for record in fold_requirements if isinstance(fold_requirements, list) else []:
            fold = record.get("fold")
            requirements = record.get("stage_requirements", {})
            for stage, value in requirements.items():
                rows.append(
                    {
                        "task": task,
                        "model": model_name,
                        "fold": fold,
                        "stage": stage,
                        "stage_requirement": value,
                    }
                )
    return pd.DataFrame(rows, columns=["task", "model", "fold", "stage", "stage_requirement"])


def run_dvr_diagnostic(task: str, run_id: str | None = None) -> DvrDiagnosticResult:
    if task not in PAPER_TASKS:
        raise ValueError(f"Unsupported DVR diagnostic task: {task}")
    run_paths = require_run(run_id=run_id)
    output_dir = run_paths.eval_dir / "dvr_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_prediction_frames(run_paths.eval_dir, task)
    progress_detail, progress_summary = _progress_at_truth_end(predictions)
    outputs = {
        f"{task}_rollout_vs_teacher_forced_metrics.csv": _rollout_metrics(predictions),
        f"{task}_error_decomposition.csv": _error_decomposition(predictions),
        f"{task}_progress_at_truth_end.csv": progress_detail,
        f"{task}_progress_at_truth_end_summary.csv": progress_summary,
        f"{task}_modifier_real_weather_daily.csv": _empty_modifier_daily(task),
        f"{task}_modifier_real_weather_summary.csv": _empty_modifier_summary(task),
        f"{task}_early_stopping_audit.csv": _early_stopping_audit(run_paths.eval_dir, task),
        f"{task}_requirement_shift_audit.csv": _requirement_shift_audit(run_paths.eval_dir, task),
    }

    output_paths: dict[str, Path] = {}
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        output_paths[filename] = path
    return DvrDiagnosticResult(task=task, output_dir=output_dir, output_paths=output_paths)
