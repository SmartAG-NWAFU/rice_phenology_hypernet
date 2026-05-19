from types import SimpleNamespace

import pandas as pd
import pytest

import rice_phenology_hypernet.cli as cli
import rice_phenology_hypernet.data.io as data_io
import rice_phenology_hypernet.experiments.dvr_diagnostic as dvr_diagnostic
import rice_phenology_hypernet.experiments.runner_dvr as runner_dvr
import rice_phenology_hypernet.features as features
from rice_phenology_hypernet.runtime import RunPaths


def test_cli_parser_exposes_public_dvr_surface_only():
    parser = cli.build_parser()
    args = parser.parse_args(["run-dvr-experiment", "--task", "sample", "--model", "m1_dvr_con"])

    assert args.command == "run-dvr-experiment"
    assert args.task == "sample"
    assert args.model == "m1_dvr_con"

    with pytest.raises(SystemExit):
        parser.parse_args(["run-dvr-experiment", "--task", "sample", "--model", "legacy_model"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run-dvr-experiment", "--task", "heldout", "--model", "m0_dvr"])


def test_cli_dispatches_public_dvr_commands(monkeypatch, tmp_path):
    calls: list[tuple[str, object]] = []
    run_paths = RunPaths(
        run_id="public_run",
        eval_dir=tmp_path / "eval",
        figures_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables",
        config_snapshot_dir=tmp_path / "config",
        manifest_path=tmp_path / "manifest.json",
    )

    monkeypatch.setattr(cli, "ensure_torch_installed", lambda: None)
    monkeypatch.setattr(data_io, "prepare_data_assets", lambda: calls.append(("prepare", None)))
    monkeypatch.setattr(features, "build_modeling_dataset", lambda force=False: calls.append(("modeling", force)) or pd.DataFrame())
    monkeypatch.setattr(features, "compute_threshold_samples", lambda force=False: calls.append(("thresholds", force)) or pd.DataFrame())
    monkeypatch.setattr("rice_phenology_hypernet.runtime.initialize_run", lambda run_id=None: run_paths)
    monkeypatch.setattr(runner_dvr, "run_dvr_experiment", lambda *args, **kwargs: calls.append(("single", (args, kwargs))))
    monkeypatch.setattr(runner_dvr, "run_all_dvr_experiments", lambda **kwargs: calls.append(("batch", kwargs)))
    monkeypatch.setattr(runner_dvr, "train_dvr_deployment_models", lambda **kwargs: calls.append(("deploy", kwargs)))
    monkeypatch.setattr(dvr_diagnostic, "run_dvr_diagnostic", lambda *args, **kwargs: calls.append(("diagnostic", (args, kwargs))))

    assert cli.main(["prepare-data"]) == 0
    assert cli.main(["invert-thresholds"]) == 0
    assert cli.main(["run-dvr-experiment", "--task", "site", "--model", "m0_dvr", "--run-id", "public_run", "--seed", "61"]) == 0
    assert cli.main(["run-all-dvr-experiments", "--run-id", "public_run", "--seeds", "61", "--device", "cpu"]) == 0
    assert cli.main(["train-dvr-deployment-models", "--run-id", "public_deploy", "--seed", "61"]) == 0
    assert cli.main(["run-dvr-diagnostic", "--task", "year", "--run-id", "public_run"]) == 0

    assert [name for name, _ in calls] == [
        "prepare",
        "modeling",
        "prepare",
        "thresholds",
        "prepare",
        "modeling",
        "single",
        "prepare",
        "modeling",
        "batch",
        "prepare",
        "modeling",
        "deploy",
        "prepare",
        "modeling",
        "diagnostic",
    ]
    assert calls[6][1][0] == ("site", "m0_dvr")
    assert calls[6][1][1]["seed"] == 61
    assert calls[9][1]["seeds"] == [61]
    assert calls[12][1] == {"run_id": "public_deploy", "seed": 61}
    assert calls[15][1][0] == ("year",)


def _prediction_frame(task: str, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SID": 1,
                "year": 2001,
                "task": task,
                "fold": 1,
                "model": model_name,
                "label": "test",
                "obs_tillering": 100.0,
                "pred_tillering": 101.0,
                "obs_jointing": 120.0,
                "pred_jointing": 119.0,
                "obs_booting": 140.0,
                "pred_booting": 142.0,
                "obs_heading": 160.0,
                "pred_heading": 159.0,
                "obs_maturity": 180.0,
                "pred_maturity": 181.0,
            }
        ]
    )


def test_run_dvr_diagnostic_writes_public_outputs(monkeypatch, tmp_path):
    eval_dir = tmp_path / "eval" / "public_run"
    eval_dir.mkdir(parents=True)
    run_paths = RunPaths(
        run_id="public_run",
        eval_dir=eval_dir,
        figures_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables",
        config_snapshot_dir=tmp_path / "config",
        manifest_path=tmp_path / "manifest.json",
    )
    for model_name in runner_dvr.PUBLIC_DVR_MODEL_NAMES:
        _prediction_frame("sample", model_name).to_csv(eval_dir / f"sample_{model_name}_predictions.csv", index=False)

    monkeypatch.setattr(dvr_diagnostic, "require_run", lambda run_id=None: run_paths)

    result = dvr_diagnostic.run_dvr_diagnostic("sample", run_id="public_run")

    expected = {
        "sample_rollout_vs_teacher_forced_metrics.csv",
        "sample_error_decomposition.csv",
        "sample_progress_at_truth_end.csv",
        "sample_progress_at_truth_end_summary.csv",
        "sample_modifier_real_weather_daily.csv",
        "sample_modifier_real_weather_summary.csv",
        "sample_early_stopping_audit.csv",
        "sample_requirement_shift_audit.csv",
    }
    assert set(result.output_paths) == expected
    for path in result.output_paths.values():
        assert path.exists()
