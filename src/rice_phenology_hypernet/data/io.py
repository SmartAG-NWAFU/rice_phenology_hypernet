from __future__ import annotations

from pathlib import Path

import pandas as pd

from rice_phenology_hypernet.types import PreparedDataPaths, RawDataPaths


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


def load_raw_weather(path: Path) -> pd.DataFrame:
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


def load_raw_phenology(path: Path) -> pd.DataFrame:
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


def prepare_data_assets(
    raw_paths: RawDataPaths,
    prepared_paths: PreparedDataPaths,
) -> PreparedDataPaths:
    """Clean raw tables and write them to explicitly supplied output paths."""

    weather = load_raw_weather(raw_paths.weather)
    phenology = load_raw_phenology(raw_paths.phenology)
    weather, phenology = _restrict_to_matching_sid_year(weather, phenology)

    for path in (
        prepared_paths.weather,
        prepared_paths.phenology,
        prepared_paths.modeling_dataset,
        prepared_paths.threshold_samples,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    weather.to_parquet(prepared_paths.weather, index=False)
    phenology.to_parquet(prepared_paths.phenology, index=False)
    return prepared_paths


def load_clean_data(
    prepared_paths: PreparedDataPaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prepared station tables without implicit preparation."""

    missing = [
        path
        for path in (prepared_paths.weather, prepared_paths.phenology)
        if not path.exists()
    ]
    if missing:
        expected = f"{prepared_paths.weather}, {prepared_paths.phenology}"
        raise FileNotFoundError(f"Prepared station data are missing; expected: {expected}")
    return (
        pd.read_parquet(prepared_paths.weather),
        pd.read_parquet(prepared_paths.phenology),
    )
