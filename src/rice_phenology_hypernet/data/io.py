from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rice_phenology_hypernet.config import get_project_config
from rice_phenology_hypernet.settings import SETTINGS
from rice_phenology_hypernet.types import PreparedDataPaths


PHENOLOGY_DATE_COLUMNS = [
    "seeding date",
    "emergence date",
    "transplanting date",
    "reviving date",
    "tillering date",
    "jointing date",
    "booting date",
    "heading date",
    "maturity date",
]

PHENOLOGY_STAGE_COLUMNS = [
    "reviving date",
    "tillering date",
    "jointing date",
    "booting date",
    "heading date",
    "maturity date",
]


def _standardize_phenology_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "station ID": "SID",
        "alt": "elevation",
        "ALT": "elevation",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in PHENOLOGY_DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["lat", "lon", "elevation", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _valid_stage_sequence(row: pd.Series) -> bool:
    last = None
    for col in PHENOLOGY_STAGE_COLUMNS:
        value = row.get(col)
        if pd.isna(value):
            continue
        if last is not None and value <= last:
            return False
        last = value
    return True


def load_raw_weather(path: Path | None = None) -> pd.DataFrame:
    config = get_project_config()
    path = path or config.data.raw_weather
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["SID", "year", "TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["SID", "Date", "TemAver"]).copy()
    if "year" not in df.columns:
        df["year"] = df["Date"].dt.year
    df["year"] = df["year"].fillna(df["Date"].dt.year).astype(int)
    return df.sort_values(["SID", "Date"]).reset_index(drop=True)


def load_raw_phenology(path: Path | None = None) -> pd.DataFrame:
    config = get_project_config()
    path = path or config.data.raw_phenology
    df = pd.read_excel(path)
    df = _standardize_phenology_columns(df)
    df = df.dropna(subset=["SID", "year", "lat", "lon", "reviving date"]).copy()
    df["year"] = df["year"].astype(int)
    valid_mask = df.apply(_valid_stage_sequence, axis=1)
    return df.loc[valid_mask].sort_values(["SID", "year"]).reset_index(drop=True)


def _restrict_to_matching_sid_year(
    weather_df: pd.DataFrame, phenology_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_keys = set(zip(weather_df["SID"], weather_df["year"]))
    pheno_keys = set(zip(phenology_df["SID"], phenology_df["year"]))
    common = weather_keys & pheno_keys
    weather_mask = weather_df.set_index(["SID", "year"]).index.isin(common)
    pheno_mask = phenology_df.set_index(["SID", "year"]).index.isin(common)
    weather_df = weather_df.loc[weather_mask].copy()
    phenology_df = phenology_df.loc[pheno_mask].copy()
    return weather_df.reset_index(drop=True), phenology_df.reset_index(drop=True)


def prepare_data_assets() -> PreparedDataPaths:
    weather = load_raw_weather()
    phenology = load_raw_phenology()
    weather, phenology = _restrict_to_matching_sid_year(weather, phenology)

    SETTINGS.processed_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.features_dir.mkdir(parents=True, exist_ok=True)
    weather_path = SETTINGS.processed_dir / "weather_clean.parquet"
    phenology_path = SETTINGS.processed_dir / "phenology_clean.parquet"
    modeling_path = SETTINGS.features_dir / "modeling_dataset.parquet"
    threshold_path = SETTINGS.features_dir / "threshold_samples.parquet"

    weather.to_parquet(weather_path, index=False)
    phenology.to_parquet(phenology_path, index=False)
    return PreparedDataPaths(
        weather=weather_path,
        phenology=phenology_path,
        modeling_dataset=modeling_path,
        threshold_samples=threshold_path,
    )


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_path = SETTINGS.processed_dir / "weather_clean.parquet"
    phenology_path = SETTINGS.processed_dir / "phenology_clean.parquet"
    if not weather_path.exists() or not phenology_path.exists():
        prepare_data_assets()
    return pd.read_parquet(weather_path), pd.read_parquet(phenology_path)
