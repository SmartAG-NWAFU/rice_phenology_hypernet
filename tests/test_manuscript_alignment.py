from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from math import exp
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest
import torch

from rice_phenology_hypernet.experiments.dvr_core import (
    DEFAULT_WEATHER_FEATURES,
    DVR_STAGE_NAMES,
    PHOTO_SENSITIVE_STAGES,
    StageInputs,
)
from rice_phenology_hypernet.experiments.dvr_summary import (
    build_dvr_relative_change_summary,
)
from rice_phenology_hypernet.experiments.regional_grid_projection import (
    REGIONAL_PERIODS,
    REGIONAL_WEATHER_DIR,
    WEATHER_SEQUENCE_LIMIT,
    _resolve_regional_periods,
)
from rice_phenology_hypernet.experiments.runner_dvr import (
    DvrWorkflowBackend,
    rollout_stage,
)
from rice_phenology_hypernet.experiments.threshold_utils import prepare_prior_map
from rice_phenology_hypernet.models.dvr_objective import (
    compute_dvr_loss,
    first_crossing_day,
)
from rice_phenology_hypernet.models.m0 import M0PhenologyModel, M0TPhenologyModel
from rice_phenology_hypernet.models.m1_dvr_con import (
    M1ConDvrConfig,
    M1ConDvrModel,
    compute_m1_dvr_con_loss,
)
from rice_phenology_hypernet.models.m1_v2_dvr import M1V2DvrConfig, M1V2DvrModel
from rice_phenology_hypernet.models.physics import (
    oryza2000_photo_response,
    trapezoidal_temperature_response,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_WEATHER_SCRIPT = run_path(
    REPOSITORY_ROOT / "scripts/meteo_download/download_regional_grid_weather_gee.py"
)
STANDARDIZE_WEATHER_SCRIPT = run_path(
    REPOSITORY_ROOT / "scripts/meteo_download/standardize_regional_grid_weather_gee.py"
)


@dataclass(frozen=True)
class FourTermLossConfig:
    event_loss_weight: float = 1.0
    terminal_loss_weight: float = 2.0
    shrink_loss_weight: float = 3.0
    smooth_loss_weight: float = 4.0
    eps: float = 1e-6


@dataclass(frozen=True)
class ConstrainedLossConfig(FourTermLossConfig):
    gate_prior_weight: float = 0.05
    gate_monotonic_weight: float = 0.05


def test_drc_objective_contains_exactly_the_four_manuscript_terms() -> None:
    outputs = {
        "completion_cdf": torch.tensor([[0.1, 0.6, 0.9]], dtype=torch.float32),
        "cum_progress_seq": torch.tensor([[0.2, 1.1, 1.4]], dtype=torch.float32),
        "log_modifier_seq": torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
    }
    mask = torch.ones((1, 3), dtype=torch.bool)

    loss, stats = compute_dvr_loss(
        outputs,
        torch.tensor([2]),
        mask,
        config=FourTermLossConfig(),
    )

    assert set(stats) == {
        "event_loss",
        "terminal_loss",
        "shrink_loss",
        "smooth_loss",
        "mae_duration",
    }
    expected = (
        stats["event_loss"]
        + 2.0 * stats["terminal_loss"]
        + 3.0 * stats["shrink_loss"]
        + 4.0 * stats["smooth_loss"]
    )
    assert float(loss) == pytest.approx(expected)


def test_cdrc_adds_only_the_two_manuscript_gate_penalties() -> None:
    model = M1ConDvrModel(
        M1ConDvrConfig(
            hidden_size=2,
            dropout=0.0,
            event_beta=12.0,
            background_gate_prior=(0.9, 0.7, 0.5, 0.3, 0.1),
        )
    )
    mask = torch.ones((1, 3), dtype=torch.bool)
    outputs = model(
        weather_seq=torch.zeros((1, 3, 5)),
        stage_state=torch.zeros((1, 2)),
        stage_index=torch.tensor([0]),
        base_dvr_seq=torch.full((1, 3), 0.4),
        mask=mask,
    )

    _, stats = compute_m1_dvr_con_loss(
        outputs,
        torch.tensor([3]),
        mask,
        model=model,
        config=ConstrainedLossConfig(),
    )

    assert set(stats) == {
        "event_loss",
        "terminal_loss",
        "shrink_loss",
        "smooth_loss",
        "mae_duration",
        "gate_prior_loss",
        "gate_monotonic_loss",
    }


def test_process_functions_and_stage_contract_match_the_manuscript() -> None:
    temperatures = np.array([8.0, 20.0, 25.0, 35.0, 38.5, 42.0])
    assert trapezoidal_temperature_response(temperatures).tolist() == pytest.approx(
        [0.0, 12.0, 17.0, 17.0, 8.5, 0.0]
    )
    assert [oryza2000_photo_response(value) for value in (12.0, 12.5, 13.5, 20.0)] == pytest.approx(
        [1.0, 1.0, 0.8, 0.0]
    )
    assert DVR_STAGE_NAMES == (
        "tillering",
        "jointing",
        "booting",
        "heading",
        "maturity",
    )
    assert PHOTO_SENSITIVE_STAGES == {"booting", "heading"}
    assert DEFAULT_WEATHER_FEATURES == (
        "TemAver",
        "TemMin",
        "TemMax",
        "daylength",
        "Precipitation",
    )


def test_regional_weather_pipeline_contains_only_manuscript_drivers() -> None:
    assert DOWNLOAD_WEATHER_SCRIPT["GEE_EXTRACT_VARS"] == [
        "temperature_2m",
        "temperature_2m_min",
        "temperature_2m_max",
        "total_precipitation_sum",
    ]
    raw = pd.DataFrame(
        {
            "point_id": ["p1"],
            "lon": [110.0],
            "lat": [30.0],
            "date": ["20030101"],
            "temperature_2m": [293.15],
            "temperature_2m_min": [288.15],
            "temperature_2m_max": [298.15],
            "total_precipitation_sum": [0.002],
        }
    )

    converted, qc = STANDARDIZE_WEATHER_SCRIPT["convert_raw_point_frame"](raw)

    assert list(converted.columns) == STANDARDIZE_WEATHER_SCRIPT["DAILY_OUTPUT_COLUMNS"]
    assert "Radiation" not in converted
    assert qc == {"clipped_negative_precipitation_values": 0}


def test_regional_weather_defaults_use_the_manuscript_period() -> None:
    assert REGIONAL_PERIODS == ("2003_2007",)
    assert _resolve_regional_periods("all") == ("2003_2007",)
    assert DOWNLOAD_WEATHER_SCRIPT["DEFAULT_START_DATE"] == "2003-01-01"
    assert DOWNLOAD_WEATHER_SCRIPT["DEFAULT_END_DATE"] == "2007-12-31"
    assert "2003_2007" in str(DOWNLOAD_WEATHER_SCRIPT["DEFAULT_POINTS_PATH"])
    assert str(DOWNLOAD_WEATHER_SCRIPT["DEFAULT_OUTPUT_DIR"]).endswith(
        "regional_grid_weather_gee_era5_2003_2007"
    )
    assert str(STANDARDIZE_WEATHER_SCRIPT["DEFAULT_INPUT_DIR"]).endswith(
        "regional_grid_weather_gee_era5_2003_2007"
    )
    assert str(STANDARDIZE_WEATHER_SCRIPT["DEFAULT_OUTPUT_DIR"]).endswith(
        "regional_grid_weather_gee_era5_2003_2007_clean"
    )
    assert str(REGIONAL_WEATHER_DIR).endswith(
        "regional_grid_weather_gee_era5_2003_2007_clean"
    )


@pytest.mark.parametrize("model_kind", ["drc", "cdrc"])
def test_learned_modifier_uses_fixed_manuscript_clip(model_kind: str) -> None:
    if model_kind == "drc":
        model = M1V2DvrModel(
            M1V2DvrConfig(hidden_size=2, dropout=0.0, event_beta=12.0)
        )
        forward_kwargs = {}
    else:
        model = M1ConDvrModel(
            M1ConDvrConfig(
                hidden_size=2,
                dropout=0.0,
                event_beta=12.0,
                background_gate_prior=(0.9, 0.7, 0.5, 0.3, 0.1),
            )
        )
        forward_kwargs = {"stage_state": torch.zeros((2, 2))}

    with torch.no_grad():
        for head in model.stage_heads:
            head.weight.zero_()
        model.stage_heads[0].bias.fill_(100.0)
        model.stage_heads[1].bias.fill_(-100.0)

    outputs = model(
        weather_seq=torch.zeros((2, 1, 5)),
        stage_index=torch.tensor([0, 1]),
        base_dvr_seq=torch.ones((2, 1)),
        mask=torch.ones((2, 1), dtype=torch.bool),
        **forward_kwargs,
    )

    assert outputs["modifier_seq"][:, 0].tolist() == pytest.approx(
        [exp(2.0), exp(-2.0)]
    )


def test_rollout_uses_120_day_final_valid_fallback() -> None:
    inputs = StageInputs(
        doy=np.arange(1.0, 151.0),
        mask=np.ones(150, dtype=bool),
        model_inputs=None,
    )

    result = rollout_stage(
        inputs=inputs,
        base_dvr=np.zeros(150),
        modifier=None,
        stage_start_doy=1.0,
    )

    assert result.completion_doy == 120.0
    assert result.next_start_doy == 121.0
    assert np.count_nonzero(result.corrected_dvr) == 0


def test_rollout_crosses_unit_progress_for_normalized_dvr() -> None:
    inputs = StageInputs(
        doy=np.arange(1.0, 4.0),
        mask=np.ones(3, dtype=bool),
        model_inputs=None,
    )

    result = rollout_stage(
        inputs=inputs,
        base_dvr=np.array([0.5, 0.5, 0.5]),
        modifier=None,
        stage_start_doy=1.0,
    )

    assert result.completion_doy == 2.0


def test_requirement_contract_identifies_the_model_family() -> None:
    parameters = signature(
        DvrWorkflowBackend.estimate_stage_requirements
    ).parameters

    assert "model_name" in parameters


def test_tensor_crossing_helper_uses_120_day_fallback() -> None:
    progress = torch.zeros((1, 150), dtype=torch.float32)
    mask = torch.ones((1, 150), dtype=torch.bool)

    assert first_crossing_day(progress, mask).item() == 120


def test_tensor_crossing_fallback_uses_the_final_valid_position() -> None:
    progress = torch.zeros((1, 3), dtype=torch.float32)
    mask = torch.tensor([[True, False, True]])

    assert first_crossing_day(progress, mask).item() == 3


def test_regional_rollout_uses_manuscript_sequence_limit() -> None:
    assert WEATHER_SEQUENCE_LIMIT == 120


def _constant_weather() -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=160, freq="D")
    return pd.DataFrame(
        {
            "SID": 1,
            "year": 2000,
            "Date": dates,
            "TemAver": 20.0,
        }
    )


def _phenology_with_missing_jointing() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SID": 1,
                "year": 2000,
                "lat": 30.0,
                "lon": 110.0,
                "elevation": 100.0,
                "transplanting date": pd.Timestamp("1999-12-27"),
                "reviving date": pd.Timestamp("2000-01-01"),
                "tillering date": pd.Timestamp("2000-01-03"),
                "jointing date": pd.NaT,
                "booting date": pd.Timestamp("2000-01-06"),
                "heading date": pd.Timestamp("2000-01-08"),
                "maturity date": pd.Timestamp("2000-01-10"),
            }
        ]
    )


@pytest.mark.parametrize(
    ("model", "collector"),
    [
        (M0TPhenologyModel(), "collect_threshold_samples_t"),
        (M0PhenologyModel(), "collect_threshold_samples"),
    ],
)
def test_threshold_samples_are_inclusive_and_retain_later_transitions(
    model: object,
    collector: str,
) -> None:
    samples = getattr(model, collector)(
        _constant_weather(),
        _phenology_with_missing_jointing(),
    )

    assert len(samples) == 1
    assert samples.loc[0, "th_reviving_tillering"] == pytest.approx(36.0)
    assert np.isnan(samples.loc[0, "th_tillering_jointing"])
    assert np.isnan(samples.loc[0, "th_jointing_booting"])
    assert samples.loc[0, "th_booting_heading"] == pytest.approx(36.0)
    assert samples.loc[0, "th_heading_maturity"] == pytest.approx(36.0)


@pytest.mark.parametrize("model", [M0TPhenologyModel(), M0PhenologyModel()])
def test_requirement_fit_keeps_the_exact_training_median(model: object) -> None:
    values = pd.DataFrame(
        {
            column: [1.234, 1.235]
            for column in model.thresholds
        }
    )
    collector_name = (
        "collect_threshold_samples_t"
        if isinstance(model, M0TPhenologyModel)
        else "collect_threshold_samples"
    )
    setattr(model, collector_name, lambda weather, phenology: values)

    fitted = model.fit(pd.DataFrame(), pd.DataFrame())

    assert list(fitted.values()) == pytest.approx([1.2345] * len(fitted))


def test_prior_map_keeps_the_exact_training_median() -> None:
    train = pd.DataFrame({"SID": [1, 2], "year": [2000, 2000]})
    thresholds = train.assign(
        **{
            column: [1.234, 1.235]
            for column in M0TPhenologyModel().thresholds
        }
    )

    prior = prepare_prior_map(train, thresholds)

    assert list(prior.values()) == pytest.approx([1.2345] * len(prior))


def test_process_prediction_uses_120_day_fallback() -> None:
    weather = _constant_weather().assign(TemAver=0.0)
    sample = pd.Series({"obs_reviving": 1.0, "latitude": 30.0})
    model = M0TPhenologyModel()
    model.thresholds = {column: 1.0 for column in model.thresholds}

    predictions = model.predict_one(weather, sample)

    assert predictions[0] == 120.0


def _write_metrics(path: Path, mae: float) -> None:
    pd.DataFrame(
        {
            "stage": [
                "tillering",
                "jointing",
                "booting",
                "heading",
                "maturity",
                "all_stage",
            ],
            "mae": mae,
            "rmse": mae + 1.0,
            "bias": mae / 2.0,
            "r2": 0.8,
        }
    ).to_csv(path, index=False)


def test_relative_summary_compares_learned_models_with_both_process_baselines(
    tmp_path: Path,
) -> None:
    for model_name, mae in {
        "m0_t": 8.0,
        "m0_dvr": 6.0,
        "m1_v2_dvr": 5.0,
        "m1_dvr_con": 4.0,
    }.items():
        _write_metrics(tmp_path / f"sample_{model_name}_metrics.csv", mae)

    output = build_dvr_relative_change_summary(
        tmp_path,
        tasks=["sample"],
    )
    summary = pd.read_csv(output)

    assert "m1_dvr_con_vs_m0_t_mae_improve_pct" in summary
    assert "m1_dvr_con_vs_m0_dvr_mae_improve_pct" in summary
    assert summary.loc[summary["stage"] == "all_stage", "m1_dvr_con_vs_m0_t_mae_improve_pct"].iloc[0] == pytest.approx(50.0)
    assert summary.loc[summary["stage"] == "all_stage", "m1_dvr_con_vs_m0_dvr_mae_improve_pct"].iloc[0] == pytest.approx(100.0 / 3.0)
