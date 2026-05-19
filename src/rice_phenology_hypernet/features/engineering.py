from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rice_phenology_hypernet.config import get_generated_feature_names, get_project_config
from rice_phenology_hypernet.data.daylength import DayLengthCalculator
from rice_phenology_hypernet.data.io import load_clean_data
from rice_phenology_hypernet.models.m0 import M0PhenologyModel
from rice_phenology_hypernet.settings import SETTINGS

THRESHOLD_COLUMNS = [
    "th_reviving_tillering",
    "th_tillering_jointing",
    "th_jointing_booting",
    "th_booting_heading",
    "th_heading_maturity",
]
THRESHOLD_STAGE_LABELS = {
    "th_reviving_tillering": "Reviving\n-> Tillering",
    "th_tillering_jointing": "Tillering\n-> Jointing",
    "th_jointing_booting": "Jointing\n-> Booting",
    "th_booting_heading": "Booting\n-> Heading",
    "th_heading_maturity": "Heading\n-> Maturity",
}
FIGURE4_STAGE_ORDER = ("reviving", "tillering", "jointing", "booting", "heading", "maturity")
FIGURE4_STAGE_LABELS = {
    "reviving": "Reviving",
    "tillering": "Tillering",
    "jointing": "Jointing",
    "booting": "Booting",
    "heading": "Heading",
    "maturity": "Maturity",
}
FIGURE4_STAGE_DOY_COLUMNS = {
    "reviving": "obs_reviving",
    "tillering": "obs_tillering",
    "jointing": "obs_jointing",
    "booting": "obs_booting",
    "heading": "obs_heading",
    "maturity": "obs_maturity",
}
THRESHOLD_DECIMALS = 2
THRESHOLD_SUMMARY_QUANTILES = {
    "q05": 0.05,
    "q25": 0.25,
    "q75": 0.75,
    "q95": 0.95,
}

FEATURE_CONFIG = get_project_config().features
WINDOWS = tuple((window.label, window.offset, window.days) for window in FEATURE_CONFIG.windows)
FINAL_FEATURES = list(get_generated_feature_names())


def required_modeling_feature_columns() -> set[str]:
    return set(get_generated_feature_names())


def threshold_stage_label(stage: str) -> str:
    return THRESHOLD_STAGE_LABELS.get(stage, stage)


def threshold_stage_labels(stages: list[str] | tuple[str, ...] | None = None) -> list[str]:
    ordered = list(stages) if stages is not None else list(THRESHOLD_COLUMNS)
    return [threshold_stage_label(stage) for stage in ordered]


def threshold_heterogeneity_stage_labels(stages: list[str] | tuple[str, ...] | None = None) -> list[str]:
    ordered = list(stages) if stages is not None else list(FIGURE4_STAGE_ORDER)
    return [FIGURE4_STAGE_LABELS.get(stage, stage) for stage in ordered]


def _threshold_value_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in THRESHOLD_COLUMNS if column in df.columns]


def build_threshold_long_form(thresholds: pd.DataFrame) -> pd.DataFrame:
    value_columns = _threshold_value_columns(thresholds)
    columns = ["SID", "year", "stage", "stage_label", "threshold"]
    if thresholds.empty or not value_columns or not {"SID", "year"}.issubset(thresholds.columns):
        return pd.DataFrame(columns=columns)

    long_df = thresholds.melt(
        id_vars=["SID", "year"],
        value_vars=value_columns,
        var_name="stage",
        value_name="threshold",
    )
    long_df["stage"] = pd.Categorical(long_df["stage"], categories=THRESHOLD_COLUMNS, ordered=True)
    long_df["stage_label"] = long_df["stage"].astype(str).map(THRESHOLD_STAGE_LABELS)
    return long_df.sort_values(["stage", "SID", "year"]).reset_index(drop=True)


def summarize_threshold_groups(
    thresholds: pd.DataFrame,
    group_col: Literal["year", "SID"],
) -> pd.DataFrame:
    if group_col not in {"year", "SID"}:
        raise ValueError(f"Unsupported group_col: {group_col}")

    count_col = "n_sites" if group_col == "year" else "n_years"
    columns = [group_col, "stage", count_col, "mean", "median", "std", "min", "q05", "q25", "q75", "q95", "max"]
    long_df = build_threshold_long_form(thresholds)
    if long_df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        long_df.groupby([group_col, "stage"], observed=True)["threshold"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            std="std",
            min="min",
            q05=lambda s: s.quantile(THRESHOLD_SUMMARY_QUANTILES["q05"]),
            q25=lambda s: s.quantile(THRESHOLD_SUMMARY_QUANTILES["q25"]),
            q75=lambda s: s.quantile(THRESHOLD_SUMMARY_QUANTILES["q75"]),
            q95=lambda s: s.quantile(THRESHOLD_SUMMARY_QUANTILES["q95"]),
            max="max",
        )
        .reset_index()
        .rename(columns={"n": count_col})
        .sort_values([group_col, "stage"])
        .reset_index(drop=True)
    )
    summary["stage"] = summary["stage"].astype(str)
    return summary[columns]


def aggregate_threshold_group_medians(
    thresholds: pd.DataFrame,
    group_col: Literal["year", "SID"],
) -> pd.DataFrame:
    if group_col not in {"year", "SID"}:
        raise ValueError(f"Unsupported group_col: {group_col}")

    value_columns = _threshold_value_columns(thresholds)
    columns = [group_col, "stage", "stage_label", "threshold"]
    if thresholds.empty or not value_columns or group_col not in thresholds.columns:
        return pd.DataFrame(columns=columns)

    aggregated = thresholds.groupby(group_col, as_index=False)[value_columns].median()
    long_df = aggregated.melt(
        id_vars=[group_col],
        value_vars=value_columns,
        var_name="stage",
        value_name="threshold",
    )
    long_df["stage"] = pd.Categorical(long_df["stage"], categories=THRESHOLD_COLUMNS, ordered=True)
    long_df["stage_label"] = long_df["stage"].astype(str).map(THRESHOLD_STAGE_LABELS)
    long_df = long_df.sort_values(["stage", group_col]).reset_index(drop=True)
    long_df["stage"] = long_df["stage"].astype(str)
    return long_df[columns]


def _build_threshold_heterogeneity_sample_base(
    thresholds: pd.DataFrame,
    modeling_df: pd.DataFrame,
) -> pd.DataFrame:
    threshold_keys = thresholds[["SID", "year", "latitude", "transplanting_doy"]].drop_duplicates().copy()
    modeling_cols = ["SID", "year", "transplanting_doy", *FIGURE4_STAGE_DOY_COLUMNS.values()]
    merged = threshold_keys.merge(
        modeling_df[modeling_cols].drop_duplicates(["SID", "year"]),
        on=["SID", "year", "transplanting_doy"],
        how="inner",
        validate="one_to_one",
    )
    for stage, doy_col in FIGURE4_STAGE_DOY_COLUMNS.items():
        merged[f"dat_{stage}"] = merged[doy_col] - merged["transplanting_doy"]
    return merged


def _compute_transplanting_based_cumulative_heat(sample_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    model = M0PhenologyModel()
    weather_index = model._build_weather_index(weather_df)
    rows: list[dict[str, float | int]] = []
    for row in sample_df.itertuples(index=False):
        key = (int(row.SID), int(row.year))
        if key not in weather_index:
            continue
        weather = model._prepare_weather(weather_index[key], float(row.latitude))
        weather["doy"] = weather["Date"].dt.dayofyear.astype(int)
        weather = weather[weather["doy"] >= int(row.transplanting_doy)].copy()
        if weather.empty:
            continue
        weather["factor"] = 1.0
        jointing_doy = getattr(row, FIGURE4_STAGE_DOY_COLUMNS["jointing"])
        heading_doy = getattr(row, FIGURE4_STAGE_DOY_COLUMNS["heading"])
        if pd.notna(jointing_doy) and pd.notna(heading_doy):
            mask = (weather["doy"] > int(jointing_doy)) & (weather["doy"] <= int(heading_doy))
            weather.loc[mask, "factor"] = weather.loc[mask, "photo"]
        weather["daily_dev"] = weather["thermal"] * weather["factor"]
        weather["cum_dev"] = weather["daily_dev"].cumsum()
        cum_map = weather.groupby("doy", sort=False)["cum_dev"].last()
        payload: dict[str, float | int] = {"SID": int(row.SID), "year": int(row.year)}
        valid = True
        prev_value = -np.inf
        for stage, doy_col in FIGURE4_STAGE_DOY_COLUMNS.items():
            stage_doy = getattr(row, doy_col)
            value = float(cum_map.get(int(stage_doy), np.nan)) if pd.notna(stage_doy) else np.nan
            payload[f"heat_{stage}"] = value
            if pd.isna(value) or value <= 0 or value < prev_value:
                valid = False
            else:
                prev_value = value
        if valid:
            rows.append(payload)
    return pd.DataFrame(rows)


def _build_threshold_heterogeneity_long_form(
    wide_df: pd.DataFrame,
    value_prefix: str,
    value_name: str,
) -> pd.DataFrame:
    value_columns = [f"{value_prefix}_{stage}" for stage in FIGURE4_STAGE_ORDER]
    long_df = wide_df.melt(
        id_vars=["SID", "year"],
        value_vars=value_columns,
        var_name="stage_key",
        value_name=value_name,
    )
    long_df["stage"] = long_df["stage_key"].str.removeprefix(f"{value_prefix}_")
    long_df["stage"] = pd.Categorical(long_df["stage"], categories=FIGURE4_STAGE_ORDER, ordered=True)
    long_df["stage_label"] = long_df["stage"].astype(str).map(FIGURE4_STAGE_LABELS)
    return long_df.sort_values(["stage", "SID", "year"]).reset_index(drop=True)[["SID", "year", "stage", "stage_label", value_name]]


def _aggregate_threshold_heterogeneity_group_medians(
    wide_df: pd.DataFrame,
    group_col: Literal["year", "SID"],
    value_prefix: str,
    value_name: str,
) -> pd.DataFrame:
    value_columns = [f"{value_prefix}_{stage}" for stage in FIGURE4_STAGE_ORDER]
    aggregated = wide_df.groupby(group_col, as_index=False)[value_columns].median()
    long_df = aggregated.melt(
        id_vars=[group_col],
        value_vars=value_columns,
        var_name="stage_key",
        value_name=value_name,
    )
    long_df["stage"] = long_df["stage_key"].str.removeprefix(f"{value_prefix}_")
    long_df["stage"] = pd.Categorical(long_df["stage"], categories=FIGURE4_STAGE_ORDER, ordered=True)
    long_df["stage_label"] = long_df["stage"].astype(str).map(FIGURE4_STAGE_LABELS)
    long_df["stage"] = long_df["stage"].astype(str)
    return long_df.sort_values(["stage", group_col]).reset_index(drop=True)[[group_col, "stage", "stage_label", value_name]]


def threshold_heterogeneity_panel_data(
    thresholds: pd.DataFrame | None = None,
    modeling_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict[str, int]]:
    threshold_df = compute_threshold_samples(force=False) if thresholds is None else thresholds.copy()
    modeling = build_modeling_dataset(force=False) if modeling_df is None else modeling_df.copy()
    if weather_df is None:
        weather, _ = load_clean_data()
    else:
        weather = weather_df.copy()

    sample_base = _build_threshold_heterogeneity_sample_base(threshold_df, modeling)
    heat_df = _compute_transplanting_based_cumulative_heat(sample_base, weather)
    sample_wide = sample_base.merge(heat_df, on=["SID", "year"], how="inner", validate="one_to_one")

    dat_long = _build_threshold_heterogeneity_long_form(sample_wide, value_prefix="dat", value_name="value")
    dat_long["metric"] = "days_after_transplanting"
    heat_long = _build_threshold_heterogeneity_long_form(sample_wide, value_prefix="heat", value_name="value")
    heat_long["metric"] = "cumulative_thermal_requirement"

    dat_spatial = _aggregate_threshold_heterogeneity_group_medians(sample_wide, group_col="SID", value_prefix="dat", value_name="value")
    dat_spatial["metric"] = "days_after_transplanting"
    heat_spatial = _aggregate_threshold_heterogeneity_group_medians(sample_wide, group_col="SID", value_prefix="heat", value_name="value")
    heat_spatial["metric"] = "cumulative_thermal_requirement"

    dat_temporal = _aggregate_threshold_heterogeneity_group_medians(sample_wide, group_col="year", value_prefix="dat", value_name="value")
    dat_temporal["metric"] = "days_after_transplanting"
    heat_temporal = _aggregate_threshold_heterogeneity_group_medians(sample_wide, group_col="year", value_prefix="heat", value_name="value")
    heat_temporal["metric"] = "cumulative_thermal_requirement"

    distribution_long = pd.concat(
        [
            dat_long.assign(scope="site_year"),
            heat_long.assign(scope="site_year"),
            dat_spatial.assign(scope="spatial"),
            heat_spatial.assign(scope="spatial"),
            dat_temporal.assign(scope="temporal"),
            heat_temporal.assign(scope="temporal"),
        ],
        ignore_index=True,
    )
    distribution_long["stage"] = pd.Categorical(distribution_long["stage"], categories=FIGURE4_STAGE_ORDER, ordered=True)
    distribution_long = distribution_long.sort_values(["metric", "scope", "stage"]).reset_index(drop=True)

    return {
        "sample_wide": sample_wide,
        "distribution_long": distribution_long,
        "counts": {
            "site_year": int(len(sample_wide)),
            "spatial": int(sample_wide["SID"].nunique()),
            "temporal": int(sample_wide["year"].nunique()),
        },
    }


def round_threshold_columns(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()
    available = [column for column in THRESHOLD_COLUMNS if column in rounded.columns]
    if available:
        rounded.loc[:, available] = rounded[available].round(THRESHOLD_DECIMALS)
    return rounded


def _build_weather_index(weather_df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    out: dict[tuple[int, int], pd.DataFrame] = {}
    for (sid, year), group in weather_df.groupby(["SID", "year"]):
        out[(int(sid), int(year))] = group.sort_values("Date").reset_index(drop=True)
    return out


def _safe_series_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def _safe_series_sum(series: pd.Series) -> float:
    return float(series.sum()) if len(series) else np.nan


def _window_slice(weather: pd.DataFrame, start: pd.Timestamp, offset: int, days: int) -> pd.DataFrame:
    begin = start + pd.Timedelta(days=offset)
    end = begin + pd.Timedelta(days=days - 1)
    return weather[(weather["Date"] >= begin) & (weather["Date"] <= end)].copy()


def _window_features(weather: pd.DataFrame, trans_date: pd.Timestamp) -> dict[str, float]:
    config = get_project_config().features
    out: dict[str, float] = {}
    for feature_name in config.explicit_feature_names:
        if feature_name.startswith("tmean_tran0_"):
            days = int(feature_name.removeprefix("tmean_tran0_"))
            out[feature_name] = _safe_series_mean(_window_slice(weather, trans_date, 0, days)["TemAver"])
            continue
        if feature_name.startswith("tmean_tran") and "_" in feature_name.removeprefix("tmean_tran"):
            bounds = feature_name.removeprefix("tmean_tran")
            start_str, end_str = bounds.split("_", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            out[feature_name] = _safe_series_mean(_window_slice(weather, trans_date, start, end - start)["TemAver"])
            continue
        raise ValueError(f"Unsupported explicit feature '{feature_name}' in features.yaml")
    for label, offset, days in WINDOWS:
        window = _window_slice(weather, trans_date, offset, days)
        if "tmean" in config.window_feature_families:
            out[f"tmean_{label}"] = _safe_series_mean(window["TemAver"])
        if "paccum" in config.window_feature_families:
            out[f"paccum_{label}"] = _safe_series_sum(window["Precipitation"])
        if "rmean" in config.window_feature_families:
            out[f"rmean_{label}"] = _safe_series_mean(window["Radiation"])
        if "hdd" in config.window_feature_families:
            out[f"hdd_{label}"] = float(
                np.clip(window["TemMax"] - config.high_temp_threshold, a_min=0.0, a_max=None).sum()
            )
        if "cdd" in config.window_feature_families:
            out[f"cdd_{label}"] = float(
                np.clip(config.low_temp_threshold - window["TemAver"], a_min=0.0, a_max=None).sum()
            )
    return out


def compute_modeling_features(
    row: pd.Series,
    weather: pd.DataFrame,
    daylength: DayLengthCalculator,
) -> dict[str, float] | None:
    sid = int(row["SID"])
    year = int(row["year"])
    trans_date = pd.to_datetime(row.get("transplanting date"), errors="coerce")
    reviving_date = pd.to_datetime(row.get("reviving date"), errors="coerce")
    seeding_date = pd.to_datetime(row.get("seeding date"), errors="coerce")
    if pd.isna(trans_date) or pd.isna(reviving_date):
        return None
    if len(weather) < 180:
        return None

    annual = weather[weather["year"] == year]
    if annual.empty:
        annual = weather

    out = {
        "SID": sid,
        "year": year,
        "latitude": float(row["lat"]),
        "longitude": float(row["lon"]),
        "altitude": float(row["elevation"]),
        "transplanting_doy": float(trans_date.dayofyear),
        "reviving_doy": float(reviving_date.dayofyear),
        "seeding_doy": _date_to_doy(seeding_date),
        "daylength_transplanting": float(
            daylength.day_length(trans_date.year, trans_date.month, trans_date.day, float(row["lat"]))
        ),
        "annual_mean_temperature": _safe_series_mean(annual["TemAver"]),
        "annual_mean_tmax": _safe_series_mean(annual["TemMax"]),
        "annual_mean_tmin": _safe_series_mean(annual["TemMin"]),
        "annual_precipitation": _safe_series_sum(annual["Precipitation"]),
        "annual_mean_radiation": _safe_series_mean(annual["Radiation"]),
        "gdd_gt_10": float(np.clip(annual["TemAver"] - FEATURE_CONFIG.temperature_base, a_min=0.0, a_max=None).sum()),
        "reviving_date": reviving_date,
        "transplanting_date": trans_date,
        "obs_reviving": float(reviving_date.dayofyear),
        "obs_tillering": _date_to_doy(row.get("tillering date")),
        "obs_jointing": _date_to_doy(row.get("jointing date")),
        "obs_booting": _date_to_doy(row.get("booting date")),
        "obs_heading": _date_to_doy(row.get("heading date")),
        "obs_maturity": _date_to_doy(row.get("maturity date")),
    }
    out.update(_window_features(weather, trans_date))
    return out


def _date_to_doy(value) -> float:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return np.nan
    return float(dt.dayofyear)


def build_modeling_dataset(force: bool = False) -> pd.DataFrame:
    output_path = SETTINGS.features_dir / "modeling_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        cached = pd.read_parquet(output_path)
        required_columns = {"SID", "year", *FINAL_FEATURES, *required_modeling_feature_columns()}
        if required_columns.issubset(cached.columns):
            return cached

    weather_df, phenology_df = load_clean_data()
    weather_index = _build_weather_index(weather_df)
    daylength = DayLengthCalculator()

    rows = []
    for _, row in phenology_df.iterrows():
        key = (int(row["SID"]), int(row["year"]))
        if key not in weather_index:
            continue
        feat = compute_modeling_features(row, weather_index[key], daylength)
        if feat is not None:
            rows.append(feat)

    dataset = pd.DataFrame(rows).sort_values(["SID", "year"]).reset_index(drop=True)
    dataset.to_parquet(output_path, index=False)
    return dataset


def compute_threshold_samples(force: bool = False) -> pd.DataFrame:
    output_path = SETTINGS.features_dir / "threshold_samples.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        thresholds = round_threshold_columns(pd.read_parquet(output_path))
        required_columns = {"SID", "year", *THRESHOLD_COLUMNS, *FINAL_FEATURES}
        if required_columns.issubset(thresholds.columns):
            thresholds.to_parquet(output_path, index=False)
            return thresholds

    weather_df, phenology_df = load_clean_data()
    model = M0PhenologyModel()
    thresholds = round_threshold_columns(model.collect_threshold_samples(weather_df, phenology_df))
    modeling = build_modeling_dataset(force=force)
    overlapping = [col for col in FINAL_FEATURES if col in thresholds.columns]
    if overlapping:
        thresholds = thresholds.drop(columns=overlapping)
    thresholds = thresholds.merge(
        modeling[["SID", "year", *FINAL_FEATURES]],
        on=["SID", "year"],
        how="left",
    )
    thresholds = round_threshold_columns(thresholds)
    thresholds.to_parquet(output_path, index=False)
    return thresholds
