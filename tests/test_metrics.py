import numpy as np
import pandas as pd

from rice_phenology_hypernet.evaluation.metrics import calculate_metrics_frame


def test_all_stage_metrics_are_stage_macro_averages():
    predictions = pd.DataFrame(
        [
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 100.0,
                "pred_tillering": 102.0,
                "obs_jointing": 120.0,
                "pred_jointing": 119.0,
                "obs_booting": 140.0,
                "pred_booting": 144.0,
                "obs_heading": 160.0,
                "pred_heading": 156.0,
                "obs_maturity": 180.0,
                "pred_maturity": 185.0,
            },
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 110.0,
                "pred_tillering": 109.0,
                "obs_jointing": 130.0,
                "pred_jointing": 132.0,
                "obs_booting": 150.0,
                "pred_booting": 149.0,
                "obs_heading": 170.0,
                "pred_heading": 173.0,
                "obs_maturity": 190.0,
                "pred_maturity": 188.0,
            }
        ]
    )

    metrics = calculate_metrics_frame(predictions)
    stage_metrics = metrics[metrics["stage"] != "all_stage"].set_index("stage")
    all_stage = metrics[metrics["stage"] == "all_stage"].iloc[0]

    assert all_stage["mae"] == stage_metrics["mae"].mean()
    assert all_stage["rmse"] == stage_metrics["rmse"].mean()
    assert all_stage["bias"] == stage_metrics["bias"].mean()
    assert all_stage["r2"] == stage_metrics["r2"].mean()
    assert all_stage["n"] == stage_metrics["n"].sum()


def test_stage_and_all_stage_r2_are_reported():
    predictions = pd.DataFrame(
        [
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 100.0,
                "pred_tillering": 102.0,
                "obs_jointing": 120.0,
                "pred_jointing": 119.0,
                "obs_booting": 140.0,
                "pred_booting": 141.0,
                "obs_heading": 160.0,
                "pred_heading": 158.0,
                "obs_maturity": 180.0,
                "pred_maturity": 183.0,
            },
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 110.0,
                "pred_tillering": 108.0,
                "obs_jointing": 130.0,
                "pred_jointing": 131.0,
                "obs_booting": 150.0,
                "pred_booting": 149.0,
                "obs_heading": 170.0,
                "pred_heading": 171.0,
                "obs_maturity": 190.0,
                "pred_maturity": 188.0,
            },
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 120.0,
                "pred_tillering": 121.0,
                "obs_jointing": 140.0,
                "pred_jointing": 138.0,
                "obs_booting": 160.0,
                "pred_booting": 162.0,
                "obs_heading": 180.0,
                "pred_heading": 179.0,
                "obs_maturity": 200.0,
                "pred_maturity": 201.0,
            },
        ]
    )

    metrics = calculate_metrics_frame(predictions)

    tillering = metrics[metrics["stage"] == "tillering"].iloc[0]
    expected_tillering_r2 = 1.0 - np.sum(np.array([2.0, -2.0, 1.0]) ** 2) / np.sum((np.array([100.0, 110.0, 120.0]) - 110.0) ** 2)
    assert tillering["r2"] == expected_tillering_r2

    all_stage = metrics[metrics["stage"] == "all_stage"].iloc[0]
    stage_r2 = metrics[metrics["stage"] != "all_stage"]["r2"]
    expected_all_stage_r2 = stage_r2.mean()
    assert all_stage["r2"] == expected_all_stage_r2


def test_r2_is_nan_when_stage_observations_have_zero_variance():
    predictions = pd.DataFrame(
        [
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 100.0,
                "pred_tillering": 101.0,
                "obs_jointing": 120.0,
                "pred_jointing": 121.0,
                "obs_booting": 140.0,
                "pred_booting": 141.0,
                "obs_heading": 160.0,
                "pred_heading": 159.0,
                "obs_maturity": 180.0,
                "pred_maturity": 181.0,
            },
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 100.0,
                "pred_tillering": 99.0,
                "obs_jointing": 130.0,
                "pred_jointing": 129.0,
                "obs_booting": 150.0,
                "pred_booting": 149.0,
                "obs_heading": 170.0,
                "pred_heading": 171.0,
                "obs_maturity": 190.0,
                "pred_maturity": 188.0,
            },
        ]
    )

    metrics = calculate_metrics_frame(predictions)

    tillering = metrics[metrics["stage"] == "tillering"].iloc[0]
    assert np.isnan(tillering["r2"])


def test_all_stage_skips_missing_stages_but_keeps_total_effective_n():
    predictions = pd.DataFrame(
        [
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 100.0,
                "pred_tillering": 101.0,
                "obs_jointing": 120.0,
                "pred_jointing": 121.0,
                "obs_booting": np.nan,
                "pred_booting": np.nan,
                "obs_heading": 160.0,
                "pred_heading": 158.0,
                "obs_maturity": 180.0,
                "pred_maturity": 179.0,
            },
            {
                "task": "site",
                "model": "m0",
                "obs_tillering": 110.0,
                "pred_tillering": 109.0,
                "obs_jointing": 130.0,
                "pred_jointing": 129.0,
                "obs_booting": np.nan,
                "pred_booting": np.nan,
                "obs_heading": 170.0,
                "pred_heading": 171.0,
                "obs_maturity": 190.0,
                "pred_maturity": 191.0,
            },
        ]
    )

    metrics = calculate_metrics_frame(predictions)

    assert "booting" not in set(metrics["stage"])
    stage_metrics = metrics[metrics["stage"] != "all_stage"]
    all_stage = metrics[metrics["stage"] == "all_stage"].iloc[0]
    assert all_stage["mae"] == stage_metrics["mae"].mean()
    assert all_stage["rmse"] == stage_metrics["rmse"].mean()
    assert all_stage["bias"] == stage_metrics["bias"].mean()
    assert all_stage["r2"] == stage_metrics["r2"].mean()
    assert all_stage["n"] == stage_metrics["n"].sum()
