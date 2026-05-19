from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rice_phenology_hypernet.config import get_generated_feature_names, get_project_config
from rice_phenology_hypernet.data import load_clean_data
from rice_phenology_hypernet.features.engineering import (
    THRESHOLD_COLUMNS,
    compute_threshold_samples,
    summarize_threshold_groups,
)
from rice_phenology_hypernet.runtime import require_run, resolve_seed_eval_dirs


CONFIG = get_project_config()
MAIN_DVR_TASKS = ("sample", "site", "year")
MODEL_ORDER = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")
MODEL_DISPLAY = {
    "m0_t": "PBM-T",
    "m0_dvr": "PBM-PT",
    "m1_v2_dvr": "DRC",
    "m1_dvr_con": "CDRC",
}
STAGE_ORDER = ("tillering", "jointing", "booting", "heading", "maturity", "all_stage")


def _evaluation_protocol_audit_dir(eval_dir: Path) -> Path:
    return eval_dir / "evaluation_protocol_audit"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_table_1(output_dir: Path) -> Path:
    weather, phenology = load_clean_data()
    summary = pd.DataFrame(
        [
            {"item": "Raw stations", "value": int(pd.read_excel(CONFIG.data.raw_phenology)["station ID"].nunique())},
            {"item": "Effective stations", "value": int(phenology["SID"].nunique())},
            {"item": "Years", "value": f"{int(phenology['year'].min())}-{int(phenology['year'].max())}"},
            {"item": "Effective seasons", "value": int(len(phenology))},
            {"item": "Mean latitude", "value": float(phenology["lat"].mean())},
            {"item": "Mean altitude", "value": float(phenology["elevation"].mean())},
            {"item": "Weather records", "value": int(len(weather))},
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_1_data_summary.csv"
    summary.to_csv(path, index=False)
    return path


def build_table_2(output_dir: Path) -> Path:
    rows = [{"group": "Stage thermal requirements", "name": name, "definition": name.replace("th_", "").replace("_", " -> ")} for name in THRESHOLD_COLUMNS]
    feature_names = list(dict.fromkeys(get_generated_feature_names()))
    rows.extend({"group": "Candidate environmental variables", "name": feature, "definition": feature.replace("_", " ")} for feature in feature_names)
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_2_threshold_and_features.csv"
    table.to_csv(path, index=False)
    return path


def build_table_3(output_dir: Path) -> Path:
    drc_cfg = CONFIG.experiment.m1_v2_dvr
    cdrc_cfg = CONFIG.experiment.m1_dvr_con
    rows = [
        {"model": "PBM-T", "component": "Daily process", "value": "Trapezoidal temperature response + stage requirement"},
        {"model": "PBM-T", "component": "Photoperiod response", "value": "Not used"},
        {"model": "PBM-T", "component": "Stage requirements", "value": "Fold-wise training-set median of inverted requirements"},
        {"model": "PBM-PT", "component": "Daily process", "value": "Trapezoidal temperature response x Oryza2000 photoperiod response"},
        {"model": "PBM-PT", "component": "Photoperiod response", "value": "Applied to photoperiod-sensitive stages"},
        {"model": "PBM-PT", "component": "Stage requirements", "value": "Fold-wise training-set median of inverted requirements"},
        {"model": "DRC", "component": "Inputs", "value": "Daily weather sequence + PBM-PT daily DVR prior"},
        {"model": "DRC", "component": "Parameterization", "value": "Positive effective daily correction on PBM-PT DVR"},
        {"model": "DRC", "component": "Network", "value": f"Sequence model (hidden={drc_cfg.hidden_size}, dropout={drc_cfg.dropout}, max_len={drc_cfg.max_sequence_length})"},
        {
            "model": "DRC",
            "component": "Training",
            "value": (
                f"epochs={drc_cfg.epochs}, batch_size={drc_cfg.batch_size}, lr={drc_cfg.learning_rate}, "
                f"selection={drc_cfg.selection_metric}"
            ),
        },
        {"model": "CDRC", "component": "Inputs", "value": "DRC inputs + constrained stage-start context state"},
        {"model": "CDRC", "component": "Parameterization", "value": "Positive effective daily correction with constrained context dependence"},
        {"model": "CDRC", "component": "Network", "value": f"Sequence model (hidden={cdrc_cfg.hidden_size}, dropout={cdrc_cfg.dropout}, max_len={cdrc_cfg.max_sequence_length})"},
        {
            "model": "CDRC",
            "component": "Training",
            "value": (
                f"epochs={cdrc_cfg.epochs}, batch_size={cdrc_cfg.batch_size}, lr={cdrc_cfg.learning_rate}, "
                f"selection={cdrc_cfg.selection_metric}, gate_prior_weight={cdrc_cfg.gate_prior_weight}"
            ),
        },
    ]
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_3_model_configuration.csv"
    table.to_csv(path, index=False)
    return path


def build_table_4(output_dir: Path, eval_dir: Path) -> Path:
    frames = []
    run_paths = type("RunPathsShim", (), {"eval_dir": Path(eval_dir), "manifest_path": Path(eval_dir) / "run_manifest.json"})()
    for seed_dir in resolve_seed_eval_dirs(run_paths):
        for task in MAIN_DVR_TASKS:
            for model in MODEL_ORDER:
                path = seed_dir / f"{task}_{model}_metrics.csv"
                if not path.exists():
                    continue
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                frame = frame[frame["stage"].isin(STAGE_ORDER)].copy()
                if frame.empty:
                    continue
                frame["seed_dir"] = seed_dir.name
                frames.append(frame[["task", "model", "stage", "mae", "rmse", "bias", "r2", "n", "seed_dir"]])
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        table = (
            combined.groupby(["task", "stage", "model"], as_index=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", lambda s: float(pd.Series(s).std(ddof=1)) if len(s) > 1 else 0.0),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", lambda s: float(pd.Series(s).std(ddof=1)) if len(s) > 1 else 0.0),
                bias_mean=("bias", "mean"),
                bias_std=("bias", lambda s: float(pd.Series(s).std(ddof=1)) if len(s) > 1 else 0.0),
                r2_mean=("r2", "mean"),
                r2_std=("r2", lambda s: float(pd.Series(s).std(ddof=1)) if len(s) > 1 else 0.0),
                n_mean=("n", "mean"),
                n_seeds=("seed_dir", "nunique"),
            )
        )
        table["task"] = pd.Categorical(table["task"], categories=MAIN_DVR_TASKS, ordered=True)
        table["stage"] = pd.Categorical(table["stage"], categories=STAGE_ORDER, ordered=True)
        table["model"] = pd.Categorical(table["model"], categories=MODEL_ORDER, ordered=True)
        table = table.sort_values(["task", "stage", "model"]).reset_index(drop=True)
        model_codes = table["model"].astype(str)
        table["model"] = model_codes.map(MODEL_DISPLAY).fillna(model_codes)
    else:
        table = pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_4_model_performance.csv"
    table.to_csv(path, index=False)
    return path


def build_table_s4(output_dir: Path, thresholds: pd.DataFrame | None = None) -> Path:
    threshold_df = compute_threshold_samples(force=False) if thresholds is None else thresholds
    table = summarize_threshold_groups(threshold_df, group_col="year")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_s4_threshold_spatial_summary_by_year.csv"
    table.to_csv(path, index=False)
    return path


def build_table_s5(output_dir: Path, thresholds: pd.DataFrame | None = None) -> Path:
    threshold_df = compute_threshold_samples(force=False) if thresholds is None else thresholds
    table = summarize_threshold_groups(threshold_df, group_col="SID")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_s5_threshold_temporal_summary_by_site.csv"
    table.to_csv(path, index=False)
    return path


def build_table_s1(output_dir: Path, eval_dir: Path) -> Path:
    plan_dir = _evaluation_protocol_audit_dir(eval_dir)
    fold_summary = _read_csv_if_exists(plan_dir / "protocol_fold_audit.csv")
    validity = _read_csv_if_exists(plan_dir / "protocol_validity_summary.csv")
    if fold_summary.empty or validity.empty:
        raise FileNotFoundError("Missing protocol audit outputs required for table_s1.")

    summary = (
        fold_summary.groupby(["task", "protocol"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            train_n_min=("train_n", "min"),
            train_n_max=("train_n", "max"),
            raw_test_n_min=("raw_test_n", "min"),
            raw_test_n_max=("raw_test_n", "max"),
            filtered_test_n_min=("filtered_test_n", "min"),
            filtered_test_n_max=("filtered_test_n", "max"),
            filtered_test_sites_min=("filtered_test_sites", "min"),
            filtered_test_sites_max=("filtered_test_sites", "max"),
            test_year_min=("filtered_test_year_min", "min"),
            test_year_max=("filtered_test_year_max", "max"),
            dropped_n_total=("dropped_n", "sum"),
        )
        .merge(validity[["task", "protocol", "valid_folds"]], on=["task", "protocol"], how="left")
    )
    summary["train_n_range"] = summary["train_n_min"].astype(int).astype(str) + "-" + summary["train_n_max"].astype(int).astype(str)
    summary["raw_test_n_range"] = summary["raw_test_n_min"].astype(int).astype(str) + "-" + summary["raw_test_n_max"].astype(int).astype(str)
    summary["filtered_test_n_range"] = summary["filtered_test_n_min"].astype(int).astype(str) + "-" + summary["filtered_test_n_max"].astype(int).astype(str)
    summary["filtered_test_sites_range"] = summary["filtered_test_sites_min"].astype(int).astype(str) + "-" + summary["filtered_test_sites_max"].astype(int).astype(str)
    summary["test_year_range"] = summary["test_year_min"].astype(int).astype(str) + "-" + summary["test_year_max"].astype(int).astype(str)
    table = summary[
        [
            "task",
            "protocol",
            "n_folds",
            "valid_folds",
            "train_n_range",
            "raw_test_n_range",
            "filtered_test_n_range",
            "filtered_test_sites_range",
            "test_year_range",
            "dropped_n_total",
        ]
    ].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_s1_protocol_audit_summary.csv"
    table.to_csv(path, index=False)
    return path


def build_table_s2(output_dir: Path, eval_dir: Path) -> Path:
    plan_dir = _evaluation_protocol_audit_dir(eval_dir)
    candidates = _read_csv_if_exists(plan_dir / "year_window_candidate_summary.csv")
    if candidates.empty:
        raise FileNotFoundError("Missing year window candidate summary required for table_s2.")
    table = candidates.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_s2_year_window_candidate_summary.csv"
    table.to_csv(path, index=False)
    return path


def build_table_s3(output_dir: Path, eval_dir: Path) -> Path:
    plan_dir = _evaluation_protocol_audit_dir(eval_dir)
    metrics = _read_csv_if_exists(plan_dir / "model_protocol_summary.csv")
    summary_path = plan_dir / "protocol_summary.json"
    if metrics.empty or not summary_path.exists():
        raise FileNotFoundError("Missing protocol model summaries required for table_s3.")
    selection = json.loads(summary_path.read_text(encoding="utf-8"))
    latest_model_label = str(selection["paper_latest_model"]["model_label"])

    m0_rows = metrics[metrics["model"] == "m0"].rename(
        columns={
            "model_label": "baseline_model_label",
            "mean_all_stage_mae": "m0_all_stage_mae",
            "mean_all_stage_rmse": "m0_all_stage_rmse",
            "mean_all_stage_bias": "m0_all_stage_bias",
        }
    )
    latest_rows = metrics[metrics["model_label"] == latest_model_label].rename(
        columns={
            "model_label": "latest_model_label",
            "mean_all_stage_mae": "latest_all_stage_mae",
            "mean_all_stage_rmse": "latest_all_stage_rmse",
            "mean_all_stage_bias": "latest_all_stage_bias",
        }
    )
    table = latest_rows.merge(
        m0_rows[["task", "protocol", "baseline_model_label", "m0_all_stage_mae", "m0_all_stage_rmse", "m0_all_stage_bias"]],
        on=["task", "protocol"],
        how="left",
    )
    table["latest_gap_vs_m0"] = table["latest_all_stage_mae"] - table["m0_all_stage_mae"]
    table = table[
        [
            "task",
            "protocol",
            "baseline_model_label",
            "m0_all_stage_mae",
            "latest_model_label",
            "latest_all_stage_mae",
            "latest_gap_vs_m0",
        ]
    ].copy()
    for column in ["m0_all_stage_mae", "latest_all_stage_mae", "latest_gap_vs_m0"]:
        table[column] = table[column].round(3)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table_s3_protocol_model_summary.csv"
    table.to_csv(path, index=False)
    return path


def build_tables(run_id: str | None = None) -> list[Path]:
    run_paths = require_run(run_id=run_id)
    run_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    return [
        build_table_1(run_paths.tables_dir),
        build_table_2(run_paths.tables_dir),
        build_table_3(run_paths.tables_dir),
        build_table_4(run_paths.tables_dir, run_paths.eval_dir),
    ]
