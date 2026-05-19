from types import SimpleNamespace

import pandas as pd

import rice_phenology_hypernet.data.io as io_module
from rice_phenology_hypernet.features import engineering as engineering_module
from rice_phenology_hypernet.features.engineering import (
    FIGURE4_STAGE_ORDER,
    THRESHOLD_COLUMNS,
    build_modeling_dataset,
    compute_threshold_samples,
    threshold_heterogeneity_panel_data,
)
from rice_phenology_hypernet.settings import build_project_settings


def _weather_frame() -> pd.DataFrame:
    rows = []
    for day in range(1, 220):
        rows.append(
            {
                "SID": 1,
                "year": 2000,
                "Date": pd.Timestamp(2000, 1, 1) + pd.Timedelta(days=day - 1),
                "TemAver": 24.0,
                "TemMin": 18.0,
                "TemMax": 30.0,
                "Precipitation": 2.0,
                "Radiation": 12.0,
            }
        )
    return pd.DataFrame(rows)


def _phenology_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SID": 1,
                "year": 2000,
                "lat": 30.0,
                "lon": 110.0,
                "elevation": 100.0,
                "seeding date": pd.Timestamp(2000, 1, 3),
                "transplanting date": pd.Timestamp(2000, 1, 10),
                "reviving date": pd.Timestamp(2000, 1, 16),
                "tillering date": pd.Timestamp(2000, 2, 5),
                "jointing date": pd.Timestamp(2000, 3, 1),
                "booting date": pd.Timestamp(2000, 4, 1),
                "heading date": pd.Timestamp(2000, 5, 1),
                "maturity date": pd.Timestamp(2000, 6, 1),
            }
        ]
    )


def test_prepare_data_assets_standardizes_and_writes_processed_files(monkeypatch, tmp_path):
    settings = build_project_settings(tmp_path)
    weather_path = tmp_path / "raw_weather.csv"
    phenology_path = tmp_path / "raw_phenology.xlsx"
    _weather_frame().to_csv(weather_path, index=False)
    _phenology_frame().rename(columns={"SID": "station ID"}).to_excel(phenology_path, index=False)

    config = SimpleNamespace(data=SimpleNamespace(raw_weather=weather_path, raw_phenology=phenology_path))
    monkeypatch.setattr(io_module, "SETTINGS", settings)
    monkeypatch.setattr(io_module, "get_project_config", lambda: config)

    paths = io_module.prepare_data_assets()

    assert paths.weather.exists()
    assert paths.phenology.exists()
    weather = pd.read_parquet(paths.weather)
    phenology = pd.read_parquet(paths.phenology)
    assert {"SID", "Date", "TemAver", "year"}.issubset(weather.columns)
    assert {"SID", "year", "lat", "lon", "reviving date"}.issubset(phenology.columns)


def test_build_modeling_dataset_and_threshold_samples_from_synthetic_clean_data(monkeypatch, tmp_path):
    settings = build_project_settings(tmp_path)
    monkeypatch.setattr(engineering_module, "SETTINGS", settings)
    monkeypatch.setattr(engineering_module, "load_clean_data", lambda: (_weather_frame(), _phenology_frame()))

    modeling = build_modeling_dataset(force=True)
    thresholds = compute_threshold_samples(force=True)

    assert len(modeling) == 1
    assert {"SID", "year", "transplanting_doy", "obs_heading", "obs_maturity"}.issubset(modeling.columns)
    assert len(thresholds) == 1
    assert {"SID", "year", *THRESHOLD_COLUMNS}.issubset(thresholds.columns)


def test_threshold_heterogeneity_panel_data_uses_site_year_scope():
    thresholds = pd.DataFrame(
        [
            {"SID": 1, "year": 2000, "latitude": 30.0, "transplanting_doy": 1.0},
            {"SID": 2, "year": 2000, "latitude": 31.0, "transplanting_doy": 1.0},
        ]
    )
    modeling = pd.DataFrame(
        [
            {
                "SID": 1,
                "year": 2000,
                "transplanting_doy": 1.0,
                "obs_reviving": 2.0,
                "obs_tillering": 3.0,
                "obs_jointing": 4.0,
                "obs_booting": 5.0,
                "obs_heading": 6.0,
                "obs_maturity": 7.0,
            },
            {
                "SID": 2,
                "year": 2000,
                "transplanting_doy": 1.0,
                "obs_reviving": 3.0,
                "obs_tillering": 4.0,
                "obs_jointing": 5.0,
                "obs_booting": 6.0,
                "obs_heading": 7.0,
                "obs_maturity": 8.0,
            },
        ]
    )
    weather = pd.DataFrame(
        [
            {"SID": sid, "year": 2000, "Date": pd.Timestamp(2000, 1, day), "TemAver": 20.0}
            for sid in (1, 2)
            for day in range(1, 11)
        ]
    )

    panel_data = threshold_heterogeneity_panel_data(thresholds=thresholds, modeling_df=modeling, weather_df=weather)

    assert panel_data["counts"]["site_year"] == 2
    assert "site_year" in set(panel_data["distribution_long"]["scope"])
    assert tuple(panel_data["distribution_long"]["stage"].drop_duplicates().tolist()) == FIGURE4_STAGE_ORDER
