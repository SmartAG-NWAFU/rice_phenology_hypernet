from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import rice_phenology_hypernet.experiments.modifier_interpretability as modifier_interpretability
from rice_phenology_hypernet.runtime import RunPaths
from rice_phenology_hypernet.experiments.modifier_interpretability import (
    _apply_perturbation,
    _dvr_star_sequence,
    _evaluate_sample_perturbations,
    _summarize_modifier_perturbations,
    analyze_modifier_interpretability,
)


class FakeModifierModel(torch.nn.Module):
    def forward(self, *, weather_seq, stage_state, stage_index, base_dvr_seq, mask):
        del stage_state, stage_index, base_dvr_seq
        log_modifier = 0.01 * weather_seq[:, :, 0] + 0.02 * weather_seq[:, :, 3] + 0.03 * weather_seq[:, :, 4]
        return {"log_modifier_seq": torch.where(mask, log_modifier, torch.zeros_like(log_modifier))}


def _weather_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TemAver": [24.0, 25.0],
            "TemMin": [20.0, 21.0],
            "TemMax": [28.0, 29.0],
            "daylength": [12.0, 12.5],
            "Precipitation": [2.0, 4.0],
        }
    )


def _sample() -> dict[str, object]:
    return _sample_for_stage("heading", 3)


def _sample_for_stage(stage: str, stage_index: int) -> dict[str, object]:
    return {
        "sid": stage_index + 1,
        "year": 2000,
        "stage_index": stage_index,
        "stage_name": stage,
        "start_doy": 180,
        "end_doy": 195,
        "true_duration": 2,
        "weather_seq": torch.tensor(_weather_frame().to_numpy(dtype=np.float32), dtype=torch.float32),
        "stage_state": torch.tensor([180.0, 50.0], dtype=torch.float32),
    }


def test_apply_perturbation_changes_only_target_input_group():
    weather = _weather_frame()

    warmed = _apply_perturbation(weather, "temperature", 3.0)
    assert list(warmed["TemAver"]) == [27.0, 28.0]
    assert list(warmed["TemMin"]) == [23.0, 24.0]
    assert list(warmed["TemMax"]) == [31.0, 32.0]
    assert list(warmed["daylength"]) == list(weather["daylength"])
    assert list(warmed["Precipitation"]) == list(weather["Precipitation"])

    longer_days = _apply_perturbation(weather, "daylength", 0.5)
    assert list(longer_days["daylength"]) == [12.5, 13.0]
    assert list(longer_days["TemAver"]) == list(weather["TemAver"])
    assert list(longer_days["Precipitation"]) == list(weather["Precipitation"])

    wetter = _apply_perturbation(weather, "precipitation", 2.0)
    assert list(wetter["Precipitation"]) == [4.0, 8.0]
    assert list(wetter["TemAver"]) == list(weather["TemAver"])
    assert list(wetter["daylength"]) == list(weather["daylength"])


def test_dvr_star_sequence_uses_base_dvr_times_modifier():
    base = np.array([0.1, 0.2], dtype=float)
    log_modifier = np.log(np.array([2.0, 0.5], dtype=float))

    dvr_star = _dvr_star_sequence(base, log_modifier)

    assert dvr_star.tolist() == pytest.approx([0.2, 0.1])


def test_evaluate_sample_perturbations_uses_original_weather_as_delta_baseline():
    rows = _evaluate_sample_perturbations(
        _sample(),
        model=FakeModifierModel(),
        stage_requirement=100.0,
        stage_label="booting_to_heading",
    )
    frame = pd.DataFrame(rows)

    temperature_zero = frame[(frame["input_group"] == "temperature") & (frame["perturbation_value"] == 0.0)].iloc[0]
    daylength_zero = frame[(frame["input_group"] == "daylength") & (frame["perturbation_value"] == 0.0)].iloc[0]
    precipitation_one = frame[(frame["input_group"] == "precipitation") & (frame["perturbation_value"] == 1.0)].iloc[0]

    assert temperature_zero["delta_log_modifier"] == pytest.approx(0.0)
    assert daylength_zero["delta_log_modifier"] == pytest.approx(0.0)
    assert precipitation_one["delta_log_modifier"] == pytest.approx(0.0)
    assert temperature_zero["delta_dvr_star_mean"] == pytest.approx(0.0)
    assert daylength_zero["delta_dvr_star_sum"] == pytest.approx(0.0)
    assert precipitation_one["delta_dvr_star_sum"] == pytest.approx(0.0)
    assert {
        "original_dvr_star_mean",
        "perturbed_dvr_star_mean",
        "delta_dvr_star_mean",
        "original_dvr_star_sum",
        "perturbed_dvr_star_sum",
        "delta_dvr_star_sum",
    }.issubset(frame.columns)
    assert frame[(frame["input_group"] == "temperature") & (frame["perturbation_value"] == 1.0)].iloc[0]["delta_log_modifier"] > 0.0
    assert frame[(frame["input_group"] == "daylength") & (frame["perturbation_value"] == 0.5)].iloc[0]["delta_log_modifier"] > 0.0
    assert frame[(frame["input_group"] == "precipitation") & (frame["perturbation_value"] == 0.0)].iloc[0]["delta_log_modifier"] < 0.0


def test_summarize_modifier_perturbations_aggregates_sample_level_deltas_with_ci():
    sample_frame = pd.DataFrame(
        [
            {
                "input_group": "temperature",
                "value_column": "temperature_offset_c",
                "perturbation_value": 0.0,
                "perturbation_unit": "deg C",
                "stage": "heading",
                "stage_label": "booting_to_heading",
                "delta_log_modifier": 0.0,
            },
            {
                "input_group": "temperature",
                "value_column": "temperature_offset_c",
                "perturbation_value": 0.0,
                "perturbation_unit": "deg C",
                "stage": "heading",
                "stage_label": "booting_to_heading",
                "delta_log_modifier": 0.2,
            },
        ]
    )

    summary = _summarize_modifier_perturbations(sample_frame, n_boot=100, random_seed=7)

    assert summary.loc[0, "delta_log_modifier_mean"] == pytest.approx(0.1)
    assert summary.loc[0, "n"] == 2
    assert {"delta_log_modifier_ci_low", "delta_log_modifier_ci_high"}.issubset(summary.columns)
    assert summary.loc[0, "delta_log_modifier_ci_low"] <= summary.loc[0, "delta_log_modifier_mean"]
    assert summary.loc[0, "delta_log_modifier_ci_high"] >= summary.loc[0, "delta_log_modifier_mean"]


def test_analyze_modifier_interpretability_writes_all_stage_outputs(monkeypatch, tmp_path):
    stages = ("tillering", "jointing", "booting", "heading", "maturity")
    run_paths = RunPaths(
        run_id="molde4_seed61",
        eval_dir=tmp_path / "eval",
        figures_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables",
        config_snapshot_dir=tmp_path / "config",
        manifest_path=tmp_path / "manifest.json",
    )

    class FakeDataset:
        def __init__(self, modeling_df, weather_df, stage_requirements):
            del modeling_df, weather_df, stage_requirements
            self.samples = [_sample_for_stage(stage, index) for index, stage in enumerate(stages)]

    monkeypatch.setattr(modifier_interpretability, "initialize_run", lambda run_id=None: run_paths)
    monkeypatch.setattr(
        modifier_interpretability,
        "load_dvr_deployment_artifact",
        lambda deployment_run_id, model_name, seed: SimpleNamespace(stage_requirements={stage: 100.0 for stage in stages}),
    )
    monkeypatch.setattr(modifier_interpretability, "materialize_dvr_deployment_artifact", lambda artifact: FakeModifierModel())
    monkeypatch.setattr(modifier_interpretability, "load_clean_data", lambda: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(modifier_interpretability, "build_modeling_dataset", lambda force=False: pd.DataFrame())
    monkeypatch.setattr(modifier_interpretability, "RiceDvrStageDataset", FakeDataset)

    result = analyze_modifier_interpretability(
        deployment_run_id="molde4_seed61",
        run_id="molde4_seed61",
        seed=61,
        n_boot=0,
        build_figure=False,
    )

    assert result.stages == stages
    assert set(result.sample_paths) == set(stages)
    assert set(result.summary_paths) == set(stages)
    assert set(result.metadata_paths) == set(stages)
    for stage in stages:
        assert result.sample_paths[stage].exists()
        assert result.summary_paths[stage].exists()
        assert result.metadata_paths[stage].exists()
