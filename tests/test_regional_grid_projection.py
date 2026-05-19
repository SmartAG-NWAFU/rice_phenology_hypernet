from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


def _cli_module():
    return importlib.import_module("rice_phenology_hypernet.cli")


def _regional_module():
    return importlib.import_module("rice_phenology_hypernet.experiments.regional_grid_projection")


def _regional_analysis_module():
    return importlib.import_module("rice_phenology_hypernet.experiments.regional_grid_analysis")


def _regional_offset_sensitivity_module():
    return importlib.import_module("rice_phenology_hypernet.experiments.regional_reviving_offset_sensitivity")


class _FakeBoundaryFrame:
    def __init__(self) -> None:
        self.plot_calls: list[dict[str, object]] = []

    @property
    def boundary(self):
        return self

    def plot(self, *args, **kwargs):
        self.plot_calls.append({"args": args, "kwargs": kwargs})
        return kwargs.get("ax")


def test_make_point_id_uses_stable_format():
    regional = _regional_module()
    assert regional.make_point_id(105.25, 28.5) == "lon_105p250000_lat_28p500000"
    assert regional.make_point_id(-105.25, -28.5) == "lon_m105p250000_lat_m28p500000"


def test_prepare_regional_grid_inputs_filters_missing_weather_and_adds_reviving(tmp_path: Path):
    regional = _regional_module()
    remote = pd.DataFrame(
        [
            {"lon": 100.0, "lat": 20.0, "transplanting_doy": 120.0, "heading_doy": 210.0, "maturity_doy": 250.0},
            {"lon": 101.0, "lat": 21.0, "transplanting_doy": 121.0, "heading_doy": 211.0, "maturity_doy": 251.0},
            {"lon": 102.0, "lat": 22.0, "transplanting_doy": 122.0, "heading_doy": 212.0, "maturity_doy": 252.0},
        ]
    )
    remote_path = tmp_path / "remote.parquet"
    remote.to_parquet(remote_path, index=False)

    valid_a = regional.make_point_id(100.0, 20.0)
    valid_b = regional.make_point_id(101.0, 21.0)
    weather_summary = pd.DataFrame(
        [
            {"point_id": valid_a, "year": 2003, "n_days": 365, "expected_days": 365},
            {"point_id": valid_a, "year": 2004, "n_days": 365, "expected_days": 365},
            {"point_id": valid_a, "year": 2008, "n_days": 365, "expected_days": 365},
            {"point_id": valid_b, "year": 2003, "n_days": 364, "expected_days": 365},
            {"point_id": valid_b, "year": 2004, "n_days": 365, "expected_days": 365},
            {"point_id": valid_b, "year": 2008, "n_days": 365, "expected_days": 365},
        ]
    )
    weather_summary_path = tmp_path / "weather_summary.parquet"
    weather_summary.to_parquet(weather_summary_path, index=False)

    result = regional.prepare_regional_grid_inputs(
        period="2003_2007",
        remote_sensing_path=remote_path,
        weather_summary_path=weather_summary_path,
        output_dir=tmp_path / "features",
    )

    valid_points = pd.read_parquet(result.valid_points_path)
    point_year_inputs = pd.read_parquet(result.point_year_inputs_path)
    excluded = pd.read_csv(result.excluded_points_path)

    assert len(valid_points) == 2
    assert len(point_year_inputs) == 4
    assert result.valid_points_path.parent.name == "2003_2007"
    assert set(point_year_inputs["period"]) == {"2003_2007"}
    assert set(point_year_inputs["year"]) == {2003, 2004}
    assert set(point_year_inputs["point_id"]) == {valid_a, valid_b}
    assert point_year_inputs.loc[point_year_inputs["point_id"] == valid_a, "obs_reviving"].iloc[0] == 125.0
    assert len(excluded) == 1
    assert excluded.iloc[0]["point_id"] == regional.make_point_id(102.0, 22.0)
    assert result.metadata["excluded_point_count"] == 1
    assert result.metadata["short_weather_point_years"] == 1
    assert result.metadata["period"] == "2003_2007"
    assert result.metadata["reviving_offset_days"] == 5.0
    assert result.metadata["reviving_rule"] == "obs_reviving = transplanting_doy + 5"

    offset_result = regional.prepare_regional_grid_inputs(
        period="2003_2007",
        remote_sensing_path=remote_path,
        weather_summary_path=weather_summary_path,
        output_dir=tmp_path / "features_offset3",
        reviving_offset_days=3.0,
    )
    offset_inputs = pd.read_parquet(offset_result.point_year_inputs_path)
    assert offset_inputs.loc[offset_inputs["point_id"] == valid_a, "obs_reviving"].iloc[0] == 123.0
    assert offset_result.metadata["reviving_offset_days"] == 3.0
    assert offset_result.metadata["reviving_rule"] == "obs_reviving = transplanting_doy + 3"


def test_prepare_regional_grid_inputs_all_uses_period_specific_remote_tables(tmp_path: Path, monkeypatch):
    regional = _regional_module()
    remote_root = tmp_path / "remote"
    point_id = regional.make_point_id(100.0, 20.0)
    for index, period in enumerate(regional.REGIONAL_PERIODS):
        period_dir = remote_root / period
        period_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "lon": 100.0,
                    "lat": 20.0,
                    "transplanting_doy": 120.0 + index,
                    "heading_doy": 210.0 + index,
                    "maturity_doy": 250.0 + index,
                }
            ]
        ).to_parquet(period_dir / regional.REMOTE_SENSING_GRID_FILENAME, index=False)

    weather_summary = pd.DataFrame(
        [
            {"point_id": point_id, "year": year, "n_days": 365, "expected_days": 365}
            for year in range(2003, 2023)
        ]
    )
    weather_summary_path = tmp_path / "weather_summary.parquet"
    weather_summary.to_parquet(weather_summary_path, index=False)

    monkeypatch.setattr(regional, "REMOTE_SENSING_GRID_DIR", remote_root)
    result = regional.prepare_regional_grid_inputs(
        period="all",
        weather_summary_path=weather_summary_path,
        output_dir=tmp_path / "features",
    )

    assert [item.period for item in result.results] == list(regional.REGIONAL_PERIODS)
    for item in result.results:
        inputs = pd.read_parquet(item.point_year_inputs_path)
        start_year, end_year = regional.REGIONAL_PERIOD_YEAR_RANGES[item.period]
        period_index = list(regional.REGIONAL_PERIODS).index(item.period)
        assert item.point_year_inputs_path.parent.name == item.period
        assert set(inputs["period"]) == {item.period}
        assert inputs["year"].min() == start_year
        assert inputs["year"].max() == end_year
        assert inputs["transplanting_doy"].iloc[0] == 120.0 + period_index


def test_run_regional_grid_projection_writes_expected_outputs(tmp_path: Path, monkeypatch):
    regional = _regional_module()
    regional_analysis = _regional_analysis_module()

    input_rows = []
    for year in (2003, 2004):
        input_rows.extend(
            [
                {
                    "point_id": "p1",
                    "year": year,
                    "lon": 100.0,
                    "lat": 20.0,
                    "transplanting_doy": 1.0,
                    "obs_reviving": 4.0,
                    "rs_heading_doy": 210.0,
                    "rs_maturity_doy": 250.0,
                },
                {
                    "point_id": "p2",
                    "year": year,
                    "lon": 101.0,
                    "lat": 21.0,
                    "transplanting_doy": 2.0,
                    "obs_reviving": 5.0,
                    "rs_heading_doy": 211.0,
                    "rs_maturity_doy": 251.0,
                },
            ]
        )
    inputs = pd.DataFrame(input_rows)
    input_path = tmp_path / "regional_grid_point_year_inputs.parquet"
    inputs.to_parquet(input_path, index=False)

    weather_rows = []
    for year in (2003, 2004):
        for point_id, lat in (("p1", 20.0), ("p2", 21.0)):
            for day in range(1, 31):
                weather_rows.append(
                    {
                        "point_id": point_id,
                        "lat": lat,
                        "Date": pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day - 1),
                        "year": year,
                        "TemAver": 26.0,
                        "TemMin": 21.0,
                        "TemMax": 31.0,
                        "Precipitation": 2.0,
                    }
                )
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    pd.DataFrame(weather_rows).to_parquet(weather_dir / "regional_weather_daily_clean_shard_00.parquet", index=False)

    call_log: list[tuple[str, tuple[int, ...]]] = []

    class _FakeLearned(torch.nn.Module):
        def __init__(self, model_name: str) -> None:
            super().__init__()
            self.model_name = model_name
            self.dummy = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        def forward(self, **kwargs):
            call_log.append((self.model_name, tuple(kwargs["weather_seq"].shape)))
            if self.model_name == "m1_v2_dvr":
                assert "stage_state" not in kwargs
            else:
                assert "stage_state" in kwargs
            cum_progress = torch.cumsum(kwargs["base_dvr_seq"] * kwargs["mask"].float(), dim=1)
            return {"cum_progress_seq": cum_progress}

    def _fake_load(run_id, model_name, *, seed=11):
        del run_id, seed
        artifact_type = "process" if model_name in {"m0_t", "m0_dvr"} else "learned"
        return SimpleNamespace(
            artifact_type=artifact_type,
            model_name=model_name,
            stage_requirements={stage: 100.0 for stage in regional.PREDICTION_STAGES},
        )

    def _fake_materialize(artifact):
        if artifact.artifact_type == "process":
            return SimpleNamespace(model_name=artifact.model_name, stage_requirements=artifact.stage_requirements)
        return _FakeLearned(artifact.model_name)

    monkeypatch.setattr(regional, "load_dvr_deployment_artifact", _fake_load)
    monkeypatch.setattr(regional, "materialize_dvr_deployment_artifact", _fake_materialize)

    result = regional.run_regional_grid_projection(
        deployment_run_id="test_bundle",
        run_id="regional_smoke",
        seed=11,
        period="2003_2007",
        input_path=input_path,
        weather_dir=weather_dir,
        chunk_size=1,
        num_workers=1,
        threads_per_worker=1,
        device="cpu",
        output_dir=tmp_path / "eval" / "regional_smoke",
    )

    yearly_predictions = pd.read_parquet(result.yearly_predictions_path)

    assert len(yearly_predictions) == 16
    assert result.period == "2003_2007"
    assert result.yearly_predictions_path.parent.name == "2003_2007"
    assert set(yearly_predictions["period"]) == {"2003_2007"}
    assert set(yearly_predictions["model"]) == {"m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con"}
    assert result.metadata["yearly_prediction_rows"] == 16
    assert result.metadata["chunk_size"] == 1
    assert result.metadata["num_workers"] == 1
    assert result.metadata["threads_per_worker"] == 1
    assert result.metadata["device"] == "cpu"
    assert result.metadata["task_count"] == 2
    assert result.metadata["task_unit"] == "year_shard"
    assert result.metadata["reviving_offset_days"] == 3.0
    assert result.metadata["reviving_rule"] == "obs_reviving = transplanting_doy + 3"
    assert result.metadata["wall_clock_seconds"] >= 0.0
    assert not (tmp_path / "eval" / "regional_smoke" / "2003_2007" / regional_analysis.CLIMATOLOGY_PREDICTIONS_FILENAME).exists()
    assert not (tmp_path / "eval" / "regional_smoke" / "2003_2007" / regional_analysis.METRICS_FILENAME).exists()
    assert any(model_name == "m1_v2_dvr" for model_name, _ in call_log)
    assert any(model_name == "m1_dvr_con" for model_name, _ in call_log)

    analysis_result = regional_analysis.analyze_regional_grid_projection(
        period="2003_2007",
        yearly_predictions_path=result.yearly_predictions_path,
        output_dir=tmp_path / "analysis" / "regional_smoke",
        build_figures=False,
    )
    climatology = pd.read_parquet(analysis_result.climatology_predictions_path)
    metrics = pd.read_csv(analysis_result.metrics_path)

    assert len(climatology) == 2
    assert analysis_result.period == "2003_2007"
    assert set(climatology["period"]) == {"2003_2007"}
    assert {"m0_t_heading_doy", "m0_dvr_maturity_doy", "m1_v2_dvr_heading_doy", "m1_dvr_con_maturity_doy"} <= set(climatology.columns)
    assert len(metrics) == 8
    assert set(metrics["stage"]) == {"heading", "maturity"}
    assert analysis_result.figure_paths == ()
    assert analysis_result.metadata["yearly_prediction_rows"] == 16


def test_run_regional_grid_projection_parallel_path_matches_single_worker(tmp_path: Path, monkeypatch):
    regional = _regional_module()
    regional_analysis = _regional_analysis_module()

    inputs = pd.DataFrame(
        [
            {"point_id": "p1", "year": 2003, "lon": 100.0, "lat": 20.0, "transplanting_doy": 1.0, "obs_reviving": 6.0, "rs_heading_doy": 210.0, "rs_maturity_doy": 250.0},
            {"point_id": "p2", "year": 2003, "lon": 101.0, "lat": 21.0, "transplanting_doy": 2.0, "obs_reviving": 7.0, "rs_heading_doy": 211.0, "rs_maturity_doy": 251.0},
            {"point_id": "p1", "year": 2004, "lon": 100.0, "lat": 20.0, "transplanting_doy": 1.0, "obs_reviving": 6.0, "rs_heading_doy": 210.0, "rs_maturity_doy": 250.0},
            {"point_id": "p2", "year": 2004, "lon": 101.0, "lat": 21.0, "transplanting_doy": 2.0, "obs_reviving": 7.0, "rs_heading_doy": 211.0, "rs_maturity_doy": 251.0},
        ]
    )
    input_path = tmp_path / "regional_grid_point_year_inputs.parquet"
    inputs.to_parquet(input_path, index=False)

    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    for shard_idx in range(2):
        weather_rows = []
        for year in (2003, 2004):
            for point_id, lat in (("p1", 20.0), ("p2", 21.0)):
                for day in range(1, 31):
                    weather_rows.append(
                        {
                            "point_id": point_id,
                            "lat": lat,
                            "Date": pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day - 1),
                            "year": year,
                            "TemAver": 26.0 + shard_idx,
                            "TemMin": 21.0,
                            "TemMax": 31.0,
                            "Precipitation": 2.0,
                        }
                    )
        pd.DataFrame(weather_rows).to_parquet(weather_dir / f"regional_weather_daily_clean_shard_0{shard_idx}.parquet", index=False)

    class _FakeLearned(torch.nn.Module):
        def __init__(self, model_name: str) -> None:
            super().__init__()
            self.model_name = model_name
            self.dummy = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        def forward(self, **kwargs):
            if self.model_name == "m1_v2_dvr":
                assert "stage_state" not in kwargs
            else:
                assert "stage_state" in kwargs
            return {"cum_progress_seq": torch.cumsum(kwargs["base_dvr_seq"] * kwargs["mask"].float(), dim=1)}

    def _fake_load(run_id, model_name, *, seed=11):
        del run_id, seed
        return SimpleNamespace(
            artifact_type="process" if model_name in {"m0_t", "m0_dvr"} else "learned",
            model_name=model_name,
            stage_requirements={stage: 100.0 for stage in regional.PREDICTION_STAGES},
        )

    def _fake_materialize(artifact):
        if artifact.artifact_type == "process":
            return SimpleNamespace(model_name=artifact.model_name, stage_requirements=artifact.stage_requirements)
        return _FakeLearned(artifact.model_name)

    class _FakeExecutor:
        def __init__(self, max_workers, initializer=None, initargs=()):
            self.max_workers = max_workers
            self.initializer = initializer
            self.initargs = initargs

        def __enter__(self):
            if self.initializer is not None:
                self.initializer(*self.initargs)
            return self

        def __exit__(self, exc_type, exc, tb):
            regional._clear_worker_state()
            return False

        def map(self, fn, iterable):
            for item in iterable:
                yield fn(item)

    monkeypatch.setattr(regional, "load_dvr_deployment_artifact", _fake_load)
    monkeypatch.setattr(regional, "materialize_dvr_deployment_artifact", _fake_materialize)
    monkeypatch.setattr(regional, "ProcessPoolExecutor", _FakeExecutor)

    serial = regional.run_regional_grid_projection(
        deployment_run_id="test_bundle",
        run_id="serial",
        seed=11,
        period="2003_2007",
        input_path=input_path,
        weather_dir=weather_dir,
        chunk_size=3,
        num_workers=1,
        threads_per_worker=1,
        device="cpu",
        output_dir=tmp_path / "eval" / "serial",
    )
    parallel = regional.run_regional_grid_projection(
        deployment_run_id="test_bundle",
        run_id="parallel",
        seed=11,
        period="2003_2007",
        input_path=input_path,
        weather_dir=weather_dir,
        chunk_size=3,
        num_workers=2,
        threads_per_worker=1,
        device="cpu",
        output_dir=tmp_path / "eval" / "parallel",
    )

    serial_yearly = pd.read_parquet(serial.yearly_predictions_path).sort_values(["point_id", "year", "model"]).reset_index(drop=True)
    parallel_yearly = pd.read_parquet(parallel.yearly_predictions_path).sort_values(["point_id", "year", "model"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(serial_yearly, parallel_yearly)

    serial_analysis = regional_analysis.analyze_regional_grid_projection(
        period="2003_2007",
        yearly_predictions_path=serial.yearly_predictions_path,
        output_dir=tmp_path / "analysis" / "serial",
        build_figures=False,
    )
    parallel_analysis = regional_analysis.analyze_regional_grid_projection(
        period="2003_2007",
        yearly_predictions_path=parallel.yearly_predictions_path,
        output_dir=tmp_path / "analysis" / "parallel",
        build_figures=False,
    )

    serial_climatology = pd.read_parquet(serial_analysis.climatology_predictions_path).sort_values(["point_id"]).reset_index(drop=True)
    parallel_climatology = pd.read_parquet(parallel_analysis.climatology_predictions_path).sort_values(["point_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(serial_climatology, parallel_climatology)

    serial_metrics = pd.read_csv(serial_analysis.metrics_path).sort_values(["model", "stage"]).reset_index(drop=True)
    parallel_metrics = pd.read_csv(parallel_analysis.metrics_path).sort_values(["model", "stage"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(serial_metrics, parallel_metrics)

    assert parallel.metadata["num_workers"] == 2
    assert parallel.metadata["threads_per_worker"] == 1
    assert parallel.metadata["device"] == "cpu"
    assert parallel.metadata["task_count"] == 4
    assert parallel.metadata["chunk_size"] == 3


def test_run_regional_grid_projection_all_writes_period_outputs(tmp_path: Path, monkeypatch):
    regional = _regional_module()

    feature_dir = tmp_path / "features"
    weather_rows = []
    for period in regional.REGIONAL_PERIODS:
        year = regional.REGIONAL_PERIOD_YEAR_RANGES[period][0]
        period_dir = feature_dir / period
        period_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "point_id": "p1",
                    "period": period,
                    "year": year,
                    "lon": 100.0,
                    "lat": 20.0,
                    "transplanting_doy": 1.0 + list(regional.REGIONAL_PERIODS).index(period),
                    "obs_reviving": 6.0 + list(regional.REGIONAL_PERIODS).index(period),
                    "rs_heading_doy": 210.0,
                    "rs_maturity_doy": 250.0,
                }
            ]
        ).to_parquet(period_dir / regional.POINT_YEAR_INPUTS_FILENAME, index=False)
        for day in range(1, 31):
            weather_rows.append(
                {
                    "point_id": "p1",
                    "lat": 20.0,
                    "Date": pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day - 1),
                    "year": year,
                    "TemAver": 26.0,
                    "TemMin": 21.0,
                    "TemMax": 31.0,
                    "Precipitation": 2.0,
                }
            )
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    pd.DataFrame(weather_rows).to_parquet(weather_dir / "regional_weather_daily_clean_shard_00.parquet", index=False)

    def _fake_load(run_id, model_name, *, seed=11):
        del run_id, seed
        return SimpleNamespace(
            artifact_type="process",
            model_name=model_name,
            stage_requirements={stage: 100.0 for stage in regional.PREDICTION_STAGES},
        )

    monkeypatch.setattr(regional, "REGIONAL_GRID_FEATURE_DIR", feature_dir)
    monkeypatch.setattr(regional, "load_dvr_deployment_artifact", _fake_load)
    monkeypatch.setattr(regional, "materialize_dvr_deployment_artifact", lambda artifact: SimpleNamespace(model_name=artifact.model_name, stage_requirements=artifact.stage_requirements))

    result = regional.run_regional_grid_projection(
        deployment_run_id="test_bundle",
        run_id="regional_all",
        seed=11,
        period="all",
        weather_dir=weather_dir,
        chunk_size=2,
        num_workers=1,
        threads_per_worker=1,
        device="cpu",
        output_dir=tmp_path / "eval" / "regional_all",
    )

    assert [item.period for item in result.results] == list(regional.REGIONAL_PERIODS)
    for item in result.results:
        yearly = pd.read_parquet(item.yearly_predictions_path)
        assert item.yearly_predictions_path.parent.name == item.period
        assert set(yearly["period"]) == {item.period}
        assert set(yearly["year"]) == {regional.REGIONAL_PERIOD_YEAR_RANGES[item.period][0]}
        assert yearly["obs_reviving"].iloc[0] == 6.0 + list(regional.REGIONAL_PERIODS).index(item.period)


def test_analyze_regional_grid_projection_all_writes_period_metrics_and_figures_by_default(tmp_path: Path, monkeypatch):
    regional = _regional_module()
    regional_analysis = _regional_analysis_module()

    input_dir = tmp_path / "eval" / "regional_all"
    output_dir = tmp_path / "analysis" / "regional_all"
    figures_dir = tmp_path / "figures" / "regional_all"
    for period in regional.REGIONAL_PERIODS:
        period_dir = input_dir / period
        period_dir.mkdir(parents=True)
        period_index = list(regional.REGIONAL_PERIODS).index(period)
        rows = []
        for point_index in range(2):
            for model_name in ("m0_t", "m1_dvr_con"):
                offset = 0.5 if model_name == "m1_dvr_con" else 1.0
                rows.append(
                    {
                        "point_id": f"p{point_index}",
                        "period": period,
                        "year": regional.REGIONAL_PERIOD_YEAR_RANGES[period][0],
                        "lon": 100.0 + point_index,
                        "lat": 20.0 + point_index,
                        "transplanting_doy": 120.0 + period_index,
                        "obs_reviving": 125.0 + period_index,
                        "rs_heading_doy": 210.0 + point_index,
                        "rs_maturity_doy": 250.0 + point_index,
                        "weather_shard": "regional_weather_daily_clean_shard_00.parquet",
                        "model": model_name,
                        "pred_tillering": 130.0,
                        "pred_jointing": 170.0,
                        "pred_booting": 195.0,
                        "pred_heading": 211.0 + point_index + offset,
                        "pred_maturity": 251.0 + point_index + offset,
                    }
                )
        pd.DataFrame(rows).to_parquet(period_dir / regional.YEARLY_PREDICTIONS_FILENAME, index=False)

    saved = {}

    def _fake_save(fig, stem, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / stem
        saved[str(path)] = fig
        fig.savefig(path, dpi=100, bbox_inches="tight")
        regional_analysis.figure_builder.plt.close(fig)
        return path

    monkeypatch.setattr(regional_analysis.figure_builder, "_load_boundary", lambda: (_FakeBoundaryFrame(), _FakeBoundaryFrame()))
    monkeypatch.setattr(regional_analysis.figure_builder, "_save", _fake_save)

    result = regional_analysis.analyze_regional_grid_projection(
        period="all",
        input_dir=input_dir,
        output_dir=output_dir,
        figures_dir=figures_dir,
    )

    assert [item.period for item in result.results] == list(regional.REGIONAL_PERIODS)
    assert result.period_metrics_path == output_dir / regional_analysis.PERIOD_METRICS_FILENAME
    period_metrics = pd.read_csv(result.period_metrics_path)
    assert set(period_metrics["period"]) == set(regional.REGIONAL_PERIODS)
    assert set(period_metrics["stage"]) == {"heading", "maturity"}
    for item in result.results:
        assert item.figure_paths == (
            figures_dir / item.period / regional_analysis.REGIONAL_COMPARISON_MAP_FIGURE_FILENAME,
            figures_dir / item.period / regional_analysis.REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME,
            figures_dir / item.period / regional_analysis.REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME,
        )
        assert item.figure_paths[0].exists()
        assert item.figure_paths[1].exists()
        assert item.figure_paths[2].exists()
        assert item.climatology_predictions_path.parent.name == item.period
        assert item.metrics_path.exists()
    assert len(saved) == len(regional.REGIONAL_PERIODS) * 3


def test_run_regional_reviving_offset_sensitivity_orchestrates_offsets(tmp_path: Path, monkeypatch):
    sensitivity = _regional_offset_sensitivity_module()
    run_paths = SimpleNamespace(
        run_id="offset_run",
        eval_dir=tmp_path / "eval" / "offset_run",
        figures_dir=tmp_path / "figures" / "offset_run",
    )
    calls: list[tuple[str, dict[str, object]]] = []
    metadata_updates: list[dict[str, object]] = []

    def _fake_prepare(**kwargs):
        calls.append(("prepare", kwargs))
        offset_dir = Path(kwargs["output_dir"]) / kwargs["period"]
        offset_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = offset_dir / "regional_grid_point_year_inputs.parquet"
        inputs_path.write_text("inputs", encoding="utf-8")
        return SimpleNamespace(
            period=kwargs["period"],
            point_year_inputs_path=inputs_path,
        )

    def _fake_run(**kwargs):
        calls.append(("run", kwargs))
        output_dir = Path(kwargs["output_dir"]) / kwargs["period"]
        output_dir.mkdir(parents=True, exist_ok=True)
        yearly_path = output_dir / "regional_grid_yearly_predictions.parquet"
        yearly_path.write_text("yearly", encoding="utf-8")
        return SimpleNamespace(
            period=kwargs["period"],
            yearly_predictions_path=yearly_path,
        )

    def _fake_analyze(**kwargs):
        calls.append(("analyze", kwargs))
        output_dir = Path(kwargs["output_dir"])
        figures_dir = Path(kwargs["figures_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        offset_days = float(output_dir.parent.name.replace("offset_", ""))
        metrics_path = output_dir / "regional_grid_heading_maturity_metrics.csv"
        pd.DataFrame(
            [
                {
                    "model": "m1_dvr_con",
                    "stage": "heading",
                    "mae": offset_days,
                    "rmse": offset_days + 0.1,
                    "bias": offset_days - 5.0,
                    "r2": 0.5,
                    "n": 2,
                }
            ]
        ).to_csv(metrics_path, index=False)
        figure_paths = (
            figures_dir / "regional_rs_m1_dvr_con_heading_maturity_comparison.png",
            figures_dir / "regional_m1_dvr_con_heading_maturity_residual_diagnostics.png",
            figures_dir / "regional_m1_dvr_con_heading_maturity_scatter.png",
        )
        for path in figure_paths:
            path.write_text("figure", encoding="utf-8")
        return SimpleNamespace(
            period=kwargs["period"],
            metrics_path=metrics_path,
            figure_paths=figure_paths,
        )

    monkeypatch.setattr(sensitivity, "initialize_run", lambda run_id=None: run_paths)
    monkeypatch.setattr(sensitivity, "prepare_regional_grid_inputs", _fake_prepare)
    monkeypatch.setattr(sensitivity, "run_regional_grid_projection", _fake_run)
    monkeypatch.setattr(sensitivity, "analyze_regional_grid_projection", _fake_analyze)
    monkeypatch.setattr(
        sensitivity,
        "update_run_metadata",
        lambda run_id, **kwargs: metadata_updates.append({"run_id": run_id, **kwargs}),
    )

    result = sensitivity.run_regional_reviving_offset_sensitivity(
        deployment_run_id="test_bundle",
        run_id="offset_run",
        seed=11,
        offsets=(3.0, 4.0, 5.0, 6.0, 7.0),
        chunk_size=32,
        num_workers=2,
        threads_per_worker=1,
    )

    assert [item.reviving_offset_days for item in result.results] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert result.summary_metrics_path == run_paths.eval_dir / sensitivity.SENSITIVITY_SUBDIR / sensitivity.SUMMARY_METRICS_FILENAME
    summary = pd.read_csv(result.summary_metrics_path)
    assert list(summary["reviving_offset_days"]) == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert set(summary["reviving_rule"]) == {
        "obs_reviving = transplanting_doy + 3",
        "obs_reviving = transplanting_doy + 4",
        "obs_reviving = transplanting_doy + 5",
        "obs_reviving = transplanting_doy + 6",
        "obs_reviving = transplanting_doy + 7",
    }
    assert set(summary["period"]) == {"2003_2007"}
    assert [name for name, _ in calls].count("prepare") == 5
    assert [name for name, _ in calls].count("run") == 5
    assert [name for name, _ in calls].count("analyze") == 5
    analyze_calls = [kwargs for name, kwargs in calls if name == "analyze"]
    assert all(kwargs["build_figures"] is True for kwargs in analyze_calls)
    assert all(Path(kwargs["figures_dir"]).parent.name.startswith("offset_") for kwargs in analyze_calls)
    prepare_offsets = [kwargs["reviving_offset_days"] for name, kwargs in calls if name == "prepare"]
    assert prepare_offsets == [3.0, 4.0, 5.0, 6.0, 7.0]
    run_devices = [kwargs["device"] for name, kwargs in calls if name == "run"]
    assert run_devices == ["cpu"] * 5
    assert metadata_updates[0]["run_id"] == "offset_run"
    assert "regional_reviving_offset_sensitivity" in metadata_updates[0]


def test_run_regional_reviving_offset_sensitivity_skips_existing_offsets(tmp_path: Path, monkeypatch):
    sensitivity = _regional_offset_sensitivity_module()
    run_paths = SimpleNamespace(
        run_id="offset_run",
        eval_dir=tmp_path / "eval" / "offset_run",
        figures_dir=tmp_path / "figures" / "offset_run",
    )
    target_dir = run_paths.eval_dir / sensitivity.SENSITIVITY_SUBDIR
    existing_analysis_dir = target_dir / "offset_4" / "analysis"
    existing_figures_dir = target_dir / "offset_4" / "figures"
    existing_analysis_dir.mkdir(parents=True)
    existing_figures_dir.mkdir(parents=True)
    existing_metrics_path = existing_analysis_dir / "regional_grid_heading_maturity_metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "m1_dvr_con",
                "stage": "heading",
                "mae": 4.0,
                "rmse": 4.1,
                "bias": -1.0,
                "r2": 0.4,
                "n": 2,
            }
        ]
    ).to_csv(existing_metrics_path, index=False)
    existing_figure = existing_figures_dir / sensitivity.REGIONAL_COMPARISON_MAP_FIGURE_FILENAME
    existing_figure.write_text("figure", encoding="utf-8")

    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_prepare(**kwargs):
        calls.append(("prepare", kwargs))
        offset_dir = Path(kwargs["output_dir"]) / kwargs["period"]
        offset_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = offset_dir / "regional_grid_point_year_inputs.parquet"
        inputs_path.write_text("inputs", encoding="utf-8")
        return SimpleNamespace(period=kwargs["period"], point_year_inputs_path=inputs_path)

    def _fake_run(**kwargs):
        calls.append(("run", kwargs))
        output_dir = Path(kwargs["output_dir"]) / kwargs["period"]
        output_dir.mkdir(parents=True, exist_ok=True)
        yearly_path = output_dir / "regional_grid_yearly_predictions.parquet"
        yearly_path.write_text("yearly", encoding="utf-8")
        return SimpleNamespace(period=kwargs["period"], yearly_predictions_path=yearly_path)

    def _fake_analyze(**kwargs):
        calls.append(("analyze", kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "regional_grid_heading_maturity_metrics.csv"
        pd.DataFrame(
            [
                {
                    "model": "m1_dvr_con",
                    "stage": "heading",
                    "mae": 3.0,
                    "rmse": 3.1,
                    "bias": -2.0,
                    "r2": 0.3,
                    "n": 2,
                }
            ]
        ).to_csv(metrics_path, index=False)
        return SimpleNamespace(period=kwargs["period"], metrics_path=metrics_path, figure_paths=())

    monkeypatch.setattr(sensitivity, "initialize_run", lambda run_id=None: run_paths)
    monkeypatch.setattr(sensitivity, "prepare_regional_grid_inputs", _fake_prepare)
    monkeypatch.setattr(sensitivity, "run_regional_grid_projection", _fake_run)
    monkeypatch.setattr(sensitivity, "analyze_regional_grid_projection", _fake_analyze)
    monkeypatch.setattr(sensitivity, "update_run_metadata", lambda run_id, **kwargs: None)

    result = sensitivity.run_regional_reviving_offset_sensitivity(
        deployment_run_id="test_bundle",
        run_id="offset_run",
        seed=11,
        offsets=(3.0, 4.0),
        num_workers=2,
    )

    assert [name for name, _ in calls] == ["prepare", "run", "analyze"]
    assert result.results[0].reviving_offset_days == 3.0
    assert result.results[0].skipped_existing is False
    assert result.results[1].reviving_offset_days == 4.0
    assert result.results[1].skipped_existing is True
    assert result.results[1].metrics_path == existing_metrics_path
    assert result.results[1].figure_paths == (existing_figure,)
    summary = pd.read_csv(result.summary_metrics_path)
    assert list(summary["reviving_offset_days"]) == [3.0, 4.0]
    assert set(summary["reviving_rule"]) == {
        "obs_reviving = transplanting_doy + 3",
        "obs_reviving = transplanting_doy + 4",
    }


def test_build_regional_grid_figures_creates_requested_layouts(tmp_path: Path, monkeypatch):
    regional_analysis = _regional_analysis_module()

    rows = []
    rng = np.random.default_rng(7)
    for index in range(18):
        lon = 100.0 + 0.3 * index
        lat = 20.0 + 0.2 * index
        rs_heading = 205.0 + index * 0.8
        rs_maturity = 245.0 + index * 0.9
        rows.append(
            {
                "point_id": f"p{index}",
                "period": "2003_2007",
                "lon": lon,
                "lat": lat,
                "transplanting_doy": 120.0 + index * 0.1,
                "obs_reviving": 125.0 + index * 0.1,
                "rs_heading_doy": rs_heading,
                "rs_maturity_doy": rs_maturity,
                "m0_t_heading_doy": rs_heading + rng.normal(1.0, 0.8),
                "m0_t_maturity_doy": rs_maturity + rng.normal(1.5, 0.9),
                "m0_dvr_heading_doy": rs_heading + rng.normal(0.5, 0.7),
                "m0_dvr_maturity_doy": rs_maturity + rng.normal(0.8, 0.8),
                "m1_v2_dvr_heading_doy": rs_heading + rng.normal(0.2, 0.5),
                "m1_v2_dvr_maturity_doy": rs_maturity + rng.normal(0.3, 0.6),
                "m1_dvr_con_heading_doy": rs_heading + rng.normal(-0.1, 0.4),
                "m1_dvr_con_maturity_doy": rs_maturity + rng.normal(0.0, 0.5),
            }
        )
    climatology_path = tmp_path / "regional_grid_climatology_predictions.parquet"
    pd.DataFrame(rows).to_parquet(climatology_path, index=False)

    saved = {}
    fake_china = _FakeBoundaryFrame()
    fake_provinces = _FakeBoundaryFrame()

    def _fake_save(fig, stem, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / stem
        saved[stem] = fig
        fig.savefig(path, dpi=100, bbox_inches="tight")
        regional_analysis.figure_builder.plt.close(fig)
        return path

    monkeypatch.setattr(regional_analysis.figure_builder, "_load_boundary", lambda: (fake_china, fake_provinces))
    monkeypatch.setattr(regional_analysis.figure_builder, "_save", _fake_save)

    result = regional_analysis.build_regional_grid_figures(
        climatology_path=climatology_path,
        output_dir=tmp_path / "figures",
    )

    assert result.map_path.name == regional_analysis.REGIONAL_COMPARISON_MAP_FIGURE_FILENAME
    assert result.map_path.exists()
    assert result.residual_diagnostic_path.name == regional_analysis.REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME
    assert result.residual_diagnostic_path.exists()
    assert result.scatter_path.name == regional_analysis.REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME
    assert result.scatter_path.exists()

    comparison_fig = saved[regional_analysis.REGIONAL_COMPARISON_MAP_FIGURE_FILENAME]
    assert len(comparison_fig.axes) == 7
    comparison_axes = comparison_fig.axes[:6]
    titles = [axis.get_title() for axis in comparison_axes]
    assert titles == [
        "Heading",
        "Maturity",
        "",
        "",
        "",
        "",
    ]
    for axis in comparison_axes:
        assert len(axis.collections) >= 1
    figure_text = "\n".join(text.get_text() for axis in comparison_axes for text in axis.texts)
    assert "Remote sensing" in figure_text
    assert "CDRC" in figure_text
    assert "DOY distribution" in figure_text
    assert "RS mean" in figure_text
    assert "Model mean" in figure_text
    assert "MAE =" not in figure_text
    assert "RMSE =" not in figure_text
    assert "Bias =" not in figure_text
    assert "R2 =" not in figure_text
    colorbar_axes = comparison_fig.axes[6:]
    assert len(colorbar_axes) == 1
    for colorbar_axis in colorbar_axes:
        bounds = colorbar_axis.get_position().bounds
        assert bounds[3] > bounds[2]

    residual_diagnostic_fig = saved[regional_analysis.REGIONAL_RESIDUAL_DIAGNOSTIC_FIGURE_FILENAME]
    assert len(residual_diagnostic_fig.axes) == 5
    residual_axes = residual_diagnostic_fig.axes[:4]
    diagnostic_titles = [axis.get_title() for axis in residual_axes]
    assert diagnostic_titles == ["Heading", "Maturity", "", ""]
    for residual_axis in residual_axes[:2]:
        residual_norm = residual_axis.collections[-1].norm
        assert getattr(residual_norm, "vcenter", None) == 0.0
        assert residual_norm.vmin == -residual_norm.vmax
    for axis in residual_axes[2:]:
        assert axis.get_xlabel() == "Prediction - remote sensing (days)"
        assert len(axis.lines) >= 3
        assert any("median" in text.get_text() for text in axis.texts)

    scatter_fig = saved[regional_analysis.REGIONAL_SCATTER_DIAGNOSTIC_FIGURE_FILENAME]
    assert len(scatter_fig.axes) == 2
    scatter_axes = scatter_fig.axes
    scatter_titles = [axis.get_title() for axis in scatter_axes]
    assert scatter_titles == ["Heading", "Maturity"]
    for axis in scatter_axes:
        assert axis.get_xlabel() == "Remote sensing DOY"
        assert axis.get_ylabel() == "CDRC DOY"
        assert len(axis.collections) >= 1
        assert len(axis.lines) >= 2
        assert any(line.get_label() == "1:1" for line in axis.lines)
        assert any(line.get_label() == "Linear fit" for line in axis.lines)
        scatter_text = "\n".join(text.get_text() for text in axis.texts)
        assert "MAE =" in scatter_text
        assert "RMSE =" in scatter_text
        assert "Bias =" in scatter_text
        assert "R2 =" in scatter_text
        assert "n =" in scatter_text
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        assert np.allclose(xlim, ylim)


def test_cli_parser_and_dispatch_cover_regional_grid_commands(tmp_path: Path, monkeypatch):
    cli = _cli_module()
    regional = _regional_module()
    regional_analysis = _regional_analysis_module()
    regional_offset_sensitivity = _regional_offset_sensitivity_module()
    parser = cli.build_parser()

    args = parser.parse_args(["prepare-regional-grid-inputs", "--period", "all", "--output-dir", str(tmp_path / "features")])
    assert args.command == "prepare-regional-grid-inputs"
    assert args.period == "all"

    args = parser.parse_args(
        [
            "run-regional-grid-projection",
            "--deployment-run-id",
            "compare_four_models_deploy_seed11",
            "--run-id",
            "regional_smoke",
            "--period",
            "all",
            "--chunk-size",
            "64",
            "--num-workers",
            "2",
            "--threads-per-worker",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert args.command == "run-regional-grid-projection"
    assert args.period == "all"
    assert args.chunk_size == 64
    assert args.num_workers == 2
    assert args.threads_per_worker == 1
    assert args.device == "cpu"

    args = parser.parse_args(["build-regional-grid-figures", "--run-id", "regional_smoke", "--period", "2008_2012"])
    assert args.command == "build-regional-grid-figures"
    assert args.period == "2008_2012"

    args = parser.parse_args(
        [
            "analyze-regional-grid-projection",
            "--run-id",
            "regional_smoke",
            "--period",
            "all",
            "--input-dir",
            str(tmp_path / "eval"),
            "--output-dir",
            str(tmp_path / "analysis"),
            "--figures-dir",
            str(tmp_path / "figures"),
        ]
    )
    assert args.command == "analyze-regional-grid-projection"
    assert args.period == "all"
    assert args.build_figures is False

    args = parser.parse_args(
        [
            "run-regional-reviving-offset-sensitivity",
            "--deployment-run-id",
            "test_bundle",
            "--run-id",
            "offset_run",
            "--seed",
            "11",
            "--offsets",
            "3",
            "5",
            "7",
            "--chunk-size",
            "16",
            "--num-workers",
            "2",
            "--threads-per-worker",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert args.command == "run-regional-reviving-offset-sensitivity"
    assert args.deployment_run_id == "test_bundle"
    assert args.run_id == "offset_run"
    assert args.seed == 11
    assert args.offsets == [3.0, 5.0, 7.0]
    assert args.chunk_size == 16
    assert args.num_workers == 2
    assert args.threads_per_worker == 1
    assert args.device == "cpu"

    default_offset_args = parser.parse_args(["run-regional-reviving-offset-sensitivity"])
    assert default_offset_args.command == "run-regional-reviving-offset-sensitivity"
    assert default_offset_args.offsets is None
    assert default_offset_args.device == "cpu"

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "ensure_torch_installed", lambda: None)
    monkeypatch.setattr(
        regional,
        "prepare_regional_grid_inputs",
        lambda **kwargs: calls.append(("prepare", kwargs))
        or SimpleNamespace(
            results=(
                SimpleNamespace(
                    period="2003_2007",
                    valid_points_path=tmp_path / "valid.parquet",
                    point_year_inputs_path=tmp_path / "inputs.parquet",
                    excluded_points_path=tmp_path / "excluded.csv",
                ),
            )
        ),
    )
    monkeypatch.setattr(
        regional,
        "run_regional_grid_projection",
        lambda **kwargs: calls.append(("run", kwargs))
        or SimpleNamespace(
            results=(
                SimpleNamespace(
                    period="2003_2007",
                    yearly_predictions_path=tmp_path / "yearly.parquet",
                    metadata_path=tmp_path / "projection_metadata.json",
                ),
            )
        ),
    )
    monkeypatch.setattr(
        regional_analysis,
        "analyze_regional_grid_projection",
        lambda **kwargs: calls.append(("analyze", kwargs))
        or SimpleNamespace(
            results=(
                SimpleNamespace(
                    period="2003_2007",
                    climatology_predictions_path=tmp_path / "climatology.parquet",
                    metrics_path=tmp_path / "metrics.csv",
                    figure_paths=(),
                    metadata_path=tmp_path / "analysis_metadata.json",
                ),
            ),
            period_metrics_path=tmp_path / "period_metrics.csv",
        ),
    )
    monkeypatch.setattr(
        regional_analysis,
        "build_regional_grid_figures",
        lambda **kwargs: calls.append(("figures", kwargs))
        or SimpleNamespace(
            map_path=tmp_path / "comparison_map.png",
        ),
    )
    monkeypatch.setattr(
        regional_offset_sensitivity,
        "run_regional_reviving_offset_sensitivity",
        lambda **kwargs: calls.append(("offset_sensitivity", kwargs))
        or SimpleNamespace(
            run_id=kwargs["run_id"],
            summary_metrics_path=tmp_path / "offset_summary.csv",
            metadata_path=tmp_path / "offset_metadata.json",
            results=(
                SimpleNamespace(
                    reviving_offset_days=3.0,
                    metrics_path=tmp_path / "offset_3_metrics.csv",
                    figure_paths=(tmp_path / "offset_3_map.png",),
                ),
            ),
        ),
    )

    assert cli.main(["prepare-regional-grid-inputs", "--output-dir", str(tmp_path / "features")]) == 0
    assert cli.main(["run-regional-grid-projection", "--run-id", "regional_smoke", "--period", "all", "--chunk-size", "32", "--num-workers", "2", "--threads-per-worker", "1", "--device", "cpu"]) == 0
    assert cli.main(["analyze-regional-grid-projection", "--run-id", "regional_smoke", "--period", "all"]) == 0
    assert cli.main(["run-regional-reviving-offset-sensitivity", "--deployment-run-id", "test_bundle", "--run-id", "offset_run", "--seed", "11", "--offsets", "3", "5", "7", "--chunk-size", "16", "--num-workers", "2", "--threads-per-worker", "1", "--device", "cpu"]) == 0
    assert cli.main(["build-regional-grid-figures", "--run-id", "regional_smoke", "--period", "2008_2012"]) == 0

    assert [item[0] for item in calls] == ["prepare", "run", "analyze", "offset_sensitivity", "figures"]
    assert calls[0][1]["period"] == "2003_2022"
    assert calls[1][1]["period"] == "all"
    assert calls[1][1]["num_workers"] == 2
    assert calls[1][1]["threads_per_worker"] == 1
    assert calls[1][1]["device"] == "cpu"
    assert calls[2][1]["period"] == "all"
    assert calls[2][1]["build_figures"] is True
    assert calls[3][1]["deployment_run_id"] == "test_bundle"
    assert calls[3][1]["run_id"] == "offset_run"
    assert calls[3][1]["seed"] == 11
    assert calls[3][1]["offsets"] == [3.0, 5.0, 7.0]
    assert calls[3][1]["chunk_size"] == 16
    assert calls[3][1]["num_workers"] == 2
    assert calls[3][1]["threads_per_worker"] == 1
    assert calls[3][1]["device"] == "cpu"
    assert calls[4][1]["period"] == "2008_2012"
