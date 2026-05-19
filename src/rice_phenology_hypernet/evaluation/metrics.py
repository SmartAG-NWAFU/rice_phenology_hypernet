from __future__ import annotations

import numpy as np
import pandas as pd


STAGES = ["tillering", "jointing", "booting", "heading", "maturity"]
METRIC_COLUMNS = ["mae", "rmse", "bias", "r2", "n"]


def _metric_arrays(df: pd.DataFrame, stage: str) -> tuple[np.ndarray, np.ndarray]:
    obs = pd.to_numeric(df[f"obs_{stage}"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(df[f"pred_{stage}"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def _r2_score(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2:
        return float("nan")
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    ss_res = float(np.sum((pred - obs) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _summarize_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    diff = pred - obs
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r2": _r2_score(obs, pred),
        "n": int(len(diff)),
    }


def _aggregate_all_stage_metrics(stage_summaries: list[dict[str, float | int]]) -> dict[str, float | int]:
    stage_df = pd.DataFrame(stage_summaries)
    return {
        "mae": float(stage_df["mae"].mean()),
        "rmse": float(stage_df["rmse"].mean()),
        "bias": float(stage_df["bias"].mean()),
        "r2": float(stage_df["r2"].mean()),
        "n": int(stage_df["n"].sum()),
    }


def calculate_metrics_frame(
    predictions: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("task", "model"),
) -> pd.DataFrame:
    rows = []
    output_columns = [*group_cols, "stage", *METRIC_COLUMNS]
    if predictions.empty:
        return pd.DataFrame(columns=output_columns)

    for group_key, group in predictions.groupby(list(group_cols)):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_cols, group_key))
        stage_summaries: list[dict[str, float | int]] = []
        for stage in STAGES:
            obs, pred = _metric_arrays(group, stage)
            if len(obs) == 0:
                continue
            stage_summary = _summarize_metrics(obs, pred)
            stage_summaries.append(stage_summary)
            rows.append({**group_values, "stage": stage, **stage_summary})
        if stage_summaries:
            rows.append({**group_values, "stage": "all_stage", **_aggregate_all_stage_metrics(stage_summaries)})
    return pd.DataFrame(rows, columns=output_columns)
