from __future__ import annotations

import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely.geometry import Point

_scipy_gaussian_kde = None

from rice_phenology_hypernet.config import get_project_config
from rice_phenology_hypernet.data import load_clean_data
from rice_phenology_hypernet.data.daylength import DayLengthCalculator
from rice_phenology_hypernet.data.io import load_raw_weather
from rice_phenology_hypernet.features import build_modeling_dataset
from rice_phenology_hypernet.features.engineering import (
    FIGURE4_STAGE_ORDER,
    THRESHOLD_COLUMNS,
    compute_threshold_samples,
    threshold_heterogeneity_panel_data,
    threshold_heterogeneity_stage_labels,
)
from rice_phenology_hypernet.runtime import RunPaths, require_run, resolve_seed_eval_dirs
from rice_phenology_hypernet.settings import SETTINGS


CONFIG = get_project_config()
FIGURE_STYLE = CONFIG.figures.style
MAIN_DVR_TASKS = ("sample", "site", "year")
STAGE_ORDER = ("tillering", "jointing", "booting", "heading", "maturity")
SUMMARY_STAGE_ORDER = (*STAGE_ORDER, "all_stage")
FIGURE6_STAGE_ORDER = STAGE_ORDER
SUMMARY_METRICS = ("mae", "bias", "rmse", "r2")
DEFAULT_FIGURE_8_STAGES = ("booting", "heading", "maturity")
SUPPLEMENTAL_MODIFIER_INTERPRETABILITY_STAGES = ("tillering", "jointing")
DEFAULT_REGIONAL_GRID_FIGURE_PERIOD = "2003_2007"
DEFAULT_REGIONAL_GRID_ANALYSIS_RUN_ID = "regional_grid_periods_seed61"
REGIONAL_GRID_ANALYSIS_SUBDIR = "regional_grid_projection"
REGIONAL_GRID_CLIMATOLOGY_FILENAME = "regional_grid_climatology_predictions.parquet"
DEFAULT_REGIONAL_REVIVING_OFFSET_SENSITIVITY_RUN_ID = "regional_reviving_offset_sensitivity_seed61"
REGIONAL_REVIVING_OFFSET_SENSITIVITY_SUBDIR = "regional_reviving_offset_sensitivity"
REGIONAL_REVIVING_OFFSET_SENSITIVITY_METRICS_FILENAME = "regional_grid_reviving_offset_sensitivity_metrics.csv"
REGIONAL_REVIVING_OFFSET_SENSITIVITY_FIGURE_FILENAME = "regional_reviving_offset_sensitivity.png"
REGIONAL_REVIVING_OFFSET_REFERENCE_DAYS = 5.0
SUPPLEMENTAL_REGIONAL_MODEL_IDS = ("m0_t", "m0_dvr", "m1_v2_dvr")
FIGURE2_CLIMATE_MONTHS = tuple(range(1, 13))
FIGURE2_DAYLENGTH_MONTHS = tuple(range(1, 13))
FIGURE2_WEATHER_YEAR_RANGE = (1984, 2010)
FIGURE2_DAYLENGTH_YEAR = 2001
FIGURE2_PHENOLOGY_STAGES = (
    ("Transplanting", "transplanting date"),
    ("Reviving", "reviving date"),
    ("Tillering", "tillering date"),
    ("Jointing", "jointing date"),
    ("Booting", "booting date"),
    ("Heading", "heading date"),
    ("Maturity", "maturity date"),
)
FIGURE2_PHENOLOGY_STAGE_COLORS = tuple(
    plt.get_cmap("viridis")(value)
    for value in np.linspace(0.18, 0.86, len(FIGURE2_PHENOLOGY_STAGES))
)
MODEL_DISPLAY = {
    "m0_t": "PBM-T",
    "m0_dvr": "PBM-PT",
    "m1_v2_dvr": "DRC",
    "m1_dvr_con": "CDRC",
}

MODEL_COLORS = {
    "m0_t": "#6F78B9",
    "m0_dvr": "#81D0D6",
    "m1_v2_dvr": "#F37252",
    "m1_dvr_con": "#FDBC63",
}

TASK_DISPLAY = {
    "sample": "Sample Random",
    "site": "Site Extrapolation",
    "year": "Year Extrapolation",
}
STAGE_DISPLAY = {
    "tillering": "Tillering",
    "jointing": "Jointing",
    "booting": "Booting",
    "heading": "Heading",
    "maturity": "Maturity",
    "all_stage": "All stages",
}
STAGE_COLORS = {
    stage: color
    for stage, color in zip(
        STAGE_ORDER,
        ["#6c8ebf", "#7fb069", "#e09f3e", "#9d4edd", "#c75146"],
        strict=False,
    )
}
FOLD_COLOR_PALETTE = tuple(plt.get_cmap("tab10").colors)
REGIONAL_CLIMATOLOGY_REQUIRED_COLUMNS = (
    "lon",
    "lat",
    "TemMin_climatology",
    "TemMax_climatology",
    "TemAver_climatology",
    "Precipitation_climatology",
)
REGIONAL_POINT_YEAR_INPUT_REQUIRED_COLUMNS = (
    "point_id",
    "year",
    "period",
    "lon",
    "lat",
)
REGIONAL_WEATHER_POINT_YEAR_REQUIRED_COLUMNS = (
    "point_id",
    "year",
    "TemMin_year",
    "TemMax_year",
    "TemAver_year",
    "Precipitation_year",
)
REGIONAL_GRID_REQUIRED_COLUMNS = (
    "point_id",
    "period",
    "lon",
    "lat",
    "rs_heading_doy",
    "rs_maturity_doy",
    "m1_dvr_con_heading_doy",
    "m1_dvr_con_maturity_doy",
)
MANUSCRIPT_MAX_WIDTH_IN = 6.89
MANUSCRIPT_MAX_HEIGHT_IN = 11.69
MANUSCRIPT_FONT_SIZES = {
    "base": 7.0,
    "tick": 6.0,
    "label": 7.0,
    "panel": 8.0,
    "annotation": 6.5,
    "legend": 6.5,
}
MANUSCRIPT_FIGURE_SIZES = {
    "framework": (MANUSCRIPT_MAX_WIDTH_IN, 3.4),
    "study_map": (5.8, 5.0),
    "study_domain": (6.0, 7.2),
    "coverage": (5.2, 5.2),
    "protocols": (MANUSCRIPT_MAX_WIDTH_IN, 3.4),
    "figure_4": (MANUSCRIPT_MAX_WIDTH_IN, 4.0),
    "figure_5": (4.3, 5.0),
    "overall_performance": (4.3, 5.0),
    "figure_7": (4.75, 3.9),
    "figure_8": (6.0, 6.0),
    "three_row_bars": (5.3, 6.0),
    "obs_sim_matrix": (5.3, 6.85),
    "regional_comparison": (5.0, 6.3),
    "regional_residual": (5.8, 5.2),
    "regional_supplemental_comparison": (4.8, 8.6),
    "regional_supplemental_residual": (4.8, 6.8),
    "regional_paired": (MANUSCRIPT_MAX_WIDTH_IN, 3.6),
    "regional_reviving_offset_sensitivity": (MANUSCRIPT_MAX_WIDTH_IN, 4.8),
    "sample_distribution": (MANUSCRIPT_MAX_WIDTH_IN, 3.2),
    "threshold_heterogeneity": (MANUSCRIPT_MAX_WIDTH_IN, 5.8),
    "regional_climatology": (MANUSCRIPT_MAX_WIDTH_IN, 6.2),
}

plt.rcParams.update(
    {
        "font.size": MANUSCRIPT_FONT_SIZES["base"],
        "axes.titlesize": MANUSCRIPT_FONT_SIZES["panel"],
        "axes.labelsize": MANUSCRIPT_FONT_SIZES["label"],
        "xtick.labelsize": MANUSCRIPT_FONT_SIZES["tick"],
        "ytick.labelsize": MANUSCRIPT_FONT_SIZES["tick"],
        "legend.fontsize": MANUSCRIPT_FONT_SIZES["legend"],
        "figure.titlesize": MANUSCRIPT_FONT_SIZES["panel"],
    }
)


def _comparison_models() -> tuple[str, ...]:
    return tuple(str(model_name) for model_name in CONFIG.paper.comparison_models)


def _save(fig, stem: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / stem
    fig.savefig(path, dpi=CONFIG.figures.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _modifier_interpretability_figure_size(n_rows: int) -> tuple[float, float]:
    height = min(MANUSCRIPT_MAX_HEIGHT_IN, max(3.4, 1.8 * float(n_rows)))
    return MANUSCRIPT_MAX_WIDTH_IN, height


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _modifier_interpretability_summary_path(eval_dir: Path, *, stage: str = "heading") -> Path:
    return Path(eval_dir) / "modifier_interpretability" / f"pghm_con_{stage}_modifier_perturbation_summary.csv"


def _modifier_interpretability_sample_path(eval_dir: Path, *, stage: str = "heading") -> Path:
    return Path(eval_dir) / "modifier_interpretability" / f"pghm_con_{stage}_modifier_perturbation_samples.csv"


def _modifier_interpretability_samples_exist(eval_dir: Path, *, stages: tuple[str, ...] = DEFAULT_FIGURE_8_STAGES) -> bool:
    return all(_modifier_interpretability_sample_path(eval_dir, stage=stage).exists() for stage in stages)


def _modifier_stage_label_for_display(stage_label: str, fallback_stage: str) -> str:
    label = stage_label if stage_label else fallback_stage
    if "_to_" in label:
        return label.replace("_to_", " \u2192 ")
    return label.replace("_", " ")


def _load_boundary() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not parse column 'adcode' as JSON; leaving as string",
            category=UserWarning,
        )
        china = gpd.read_file(CONFIG.data.china_boundary)
        provinces = gpd.read_file(CONFIG.data.province_boundary)
    return china, provinces


def _regional_climatology_path() -> Path:
    data_dir = CONFIG.data.china_boundary.parents[1]
    return (
        data_dir
        / "processed"
        / "regional_grid_weather_gee_era5_2003_2022_clean"
        / "regional_weather_climatology_2003_2022.parquet"
    )


def _regional_weather_point_year_summary_path() -> Path:
    data_dir = CONFIG.data.china_boundary.parents[1]
    return (
        data_dir
        / "processed"
        / "regional_grid_weather_gee_era5_2003_2022_clean"
        / "regional_weather_point_year_summary.parquet"
    )


def _regional_point_year_inputs_path(period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD) -> Path:
    data_dir = CONFIG.data.china_boundary.parents[1]
    return (
        data_dir
        / "artifacts"
        / "features"
        / REGIONAL_GRID_ANALYSIS_SUBDIR
        / period
        / "regional_grid_point_year_inputs.parquet"
    )


def _regional_period_label(period: str) -> str:
    return period.replace("_", "-")


def _read_regional_climatology_file(path: Path) -> pd.DataFrame:
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Missing regional climatology file: {target_path}. "
            "Run scripts/meteo_download/standardize_regional_grid_weather_gee.py first."
        )

    climatology = pd.read_parquet(target_path)
    missing = [column for column in REGIONAL_CLIMATOLOGY_REQUIRED_COLUMNS if column not in climatology.columns]
    if missing:
        raise ValueError(f"Regional climatology table missing required columns: {missing}")
    return climatology.copy()


def _load_regional_climatology_table(
    path: Path | None = None,
    *,
    period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD,
    point_year_inputs_path: Path | str | None = None,
    weather_summary_path: Path | str | None = None,
) -> pd.DataFrame:
    if path is not None:
        return _read_regional_climatology_file(Path(path))

    inputs_path = (
        _regional_point_year_inputs_path(period)
        if point_year_inputs_path is None
        else Path(point_year_inputs_path)
    )
    weather_path = (
        _regional_weather_point_year_summary_path()
        if weather_summary_path is None
        else Path(weather_summary_path)
    )
    if not inputs_path.exists():
        raise FileNotFoundError(
            f"Missing regional point-year inputs for Figure S7: {inputs_path}. "
            f"Run prepare-regional-grid-inputs --period {period} first."
        )
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Missing regional weather point-year summary for Figure S7: {weather_path}. "
            "Run scripts/meteo_download/standardize_regional_grid_weather_gee.py first."
        )

    point_year_inputs = pd.read_parquet(inputs_path).copy()
    missing_inputs = [
        column for column in REGIONAL_POINT_YEAR_INPUT_REQUIRED_COLUMNS if column not in point_year_inputs.columns
    ]
    if missing_inputs:
        raise ValueError(f"Regional point-year inputs missing required columns: {missing_inputs}")

    weather_summary = pd.read_parquet(weather_path).copy()
    missing_weather = [
        column for column in REGIONAL_WEATHER_POINT_YEAR_REQUIRED_COLUMNS if column not in weather_summary.columns
    ]
    if missing_weather:
        raise ValueError(f"Regional weather point-year summary missing required columns: {missing_weather}")

    point_year_inputs["year"] = pd.to_numeric(point_year_inputs["year"], errors="raise").astype(int)
    weather_summary["year"] = pd.to_numeric(weather_summary["year"], errors="raise").astype(int)
    point_year_inputs = point_year_inputs.loc[point_year_inputs["period"].astype(str) == period].copy()
    point_year_inputs = point_year_inputs[
        ["point_id", "year", "period", "lon", "lat"]
    ].drop_duplicates(["point_id", "year"])
    if point_year_inputs.empty:
        raise ValueError(f"Regional point-year inputs contain no rows for period {period}.")

    annual_value_columns = [
        "TemMin_year",
        "TemMax_year",
        "TemAver_year",
        "Precipitation_year",
    ]
    annual_columns = [
        "point_id",
        "year",
        *annual_value_columns,
    ]
    merged = point_year_inputs.merge(
        weather_summary[annual_columns],
        on=["point_id", "year"],
        how="left",
        validate="one_to_one",
    )
    annual_missing = merged[annual_value_columns].isna().any(axis=1)
    if annual_missing.any():
        missing_count = int(annual_missing.sum())
        raise ValueError(
            f"Regional Figure S7 climatology has {missing_count} point-years without annual weather summaries "
            f"for period {period}."
        )

    climatology = (
        merged.groupby(["point_id", "period", "lon", "lat"], as_index=False)
        .agg(
            n_years_available=("year", "nunique"),
            TemMin_climatology=("TemMin_year", "mean"),
            TemMax_climatology=("TemMax_year", "mean"),
            TemAver_climatology=("TemAver_year", "mean"),
            Precipitation_climatology=("Precipitation_year", "mean"),
        )
        .sort_values(["lat", "lon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    missing = [column for column in REGIONAL_CLIMATOLOGY_REQUIRED_COLUMNS if column not in climatology.columns]
    if missing:
        raise ValueError(f"Regional climatology table missing required columns: {missing}")
    return climatology


def _regional_grid_climatology_path(run_dir: Path, *, period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD) -> Path:
    for candidate in _regional_grid_climatology_candidates(run_dir, period=period):
        if candidate.exists():
            return candidate
    return _regional_grid_climatology_candidates(run_dir, period=period)[0]


def _regional_reviving_offset_sensitivity_metrics_path(
    sensitivity_run_id: str = DEFAULT_REGIONAL_REVIVING_OFFSET_SENSITIVITY_RUN_ID,
) -> Path:
    return (
        SETTINGS.eval_dir
        / sensitivity_run_id
        / REGIONAL_REVIVING_OFFSET_SENSITIVITY_SUBDIR
        / REGIONAL_REVIVING_OFFSET_SENSITIVITY_METRICS_FILENAME
    )


def _regional_grid_climatology_candidates(run_dir: Path, *, period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD) -> list[Path]:
    run_dir = Path(run_dir)
    candidates = [
        run_dir / REGIONAL_GRID_ANALYSIS_SUBDIR / period / REGIONAL_GRID_CLIMATOLOGY_FILENAME,
    ]
    for regional_run_id in _regional_grid_analysis_run_id_candidates(run_dir):
        candidates.append(
            run_dir.parent
            / regional_run_id
            / REGIONAL_GRID_ANALYSIS_SUBDIR
            / period
            / REGIONAL_GRID_CLIMATOLOGY_FILENAME
        )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _regional_grid_analysis_run_id_candidates(run_dir: Path) -> list[str]:
    seed_match = re.search(r"seed[_-]?(\d+)", Path(run_dir).name)
    candidates = []
    if seed_match is not None:
        candidates.append(f"regional_grid_periods_seed{seed_match.group(1)}")
    candidates.append(DEFAULT_REGIONAL_GRID_ANALYSIS_RUN_ID)
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _load_regional_grid_climatology(
    run_dir: Path,
    *,
    period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD,
    climatology_path: Path | str | None = None,
) -> pd.DataFrame:
    if climatology_path is not None:
        target_path = Path(climatology_path)
        checked_paths = [target_path]
    else:
        checked_paths = _regional_grid_climatology_candidates(run_dir, period=period)
        target_path = next((candidate for candidate in checked_paths if candidate.exists()), checked_paths[0])
    if not target_path.exists():
        checked_display = "; ".join(str(path) for path in checked_paths)
        raise FileNotFoundError(
            f"Missing regional grid climatology predictions. Checked: {checked_display}. "
            f"Run analyze-regional-grid-projection --run-id {DEFAULT_REGIONAL_GRID_ANALYSIS_RUN_ID} --period {period} first, "
            "or pass climatology_path explicitly."
        )

    climatology = pd.read_parquet(target_path)
    missing = [column for column in REGIONAL_GRID_REQUIRED_COLUMNS if column not in climatology.columns]
    if missing:
        raise ValueError(f"Regional grid climatology table missing required columns: {missing}")
    return climatology.copy()


def _load_study_area_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    _, phenology = load_clean_data()
    china, provinces = _load_boundary()
    stations = phenology.drop_duplicates("SID").dropna(subset=["lon", "lat"])
    coverage = pd.crosstab(phenology["SID"], phenology["year"])
    geometry = [Point(lon, lat) for lon, lat in zip(stations["lon"], stations["lat"])]
    sites = gpd.GeoDataFrame(stations, geometry=geometry, crs="EPSG:4326")
    return china, provinces, sites, coverage


def _stage_threshold_arrays(long_df: pd.DataFrame) -> list[np.ndarray]:
    return [
        long_df.loc[long_df["stage"] == stage, "threshold"].dropna().to_numpy(dtype=float)
        for stage in THRESHOLD_COLUMNS
    ]


def _stage_value_arrays(long_df: pd.DataFrame, stage_order: tuple[str, ...], value_col: str = "value") -> list[np.ndarray]:
    return [
        long_df.loc[long_df["stage"] == stage, value_col].dropna().to_numpy(dtype=float)
        for stage in stage_order
    ]


def _style_boxplot(boxplot: dict[str, object], facecolor: str) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor(facecolor)
        box.set_alpha(0.65)
    for median in boxplot["medians"]:
        median.set_color(FIGURE_STYLE.accent)
        median.set_linewidth(1.4)
    for whisker in boxplot["whiskers"]:
        whisker.set_color(FIGURE_STYLE.accent)
    for cap in boxplot["caps"]:
        cap.set_color(FIGURE_STYLE.accent)


def _draw_distribution_panel(ax, data: list[np.ndarray], kind: str, color: str) -> None:
    has_data = any(len(values) for values in data)
    if not has_data:
        ax.text(0.5, 0.5, "No threshold data", ha="center", va="center", transform=ax.transAxes)
        return

    can_draw_violin = (
        kind == "violin"
        and all(len(values) >= 2 for values in data)
        and all(np.ptp(values) > 0 for values in data)
    )
    if can_draw_violin:
        try:
            violin = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.7)
            violin["cmedians"].set_color(FIGURE_STYLE.accent)
            violin["cmedians"].set_linewidth(1.4)
            return
        except ValueError:
            pass
        except np.linalg.LinAlgError:
            pass

    boxplot = ax.boxplot(data, patch_artist=True, showfliers=False)
    _style_boxplot(boxplot, facecolor=color)


def _coerce_run_paths(run_dir_or_paths: Path | RunPaths) -> RunPaths:
    if isinstance(run_dir_or_paths, RunPaths):
        return run_dir_or_paths
    run_dir = Path(run_dir_or_paths)
    return RunPaths(
        run_id=run_dir.name,
        eval_dir=run_dir,
        figures_dir=run_dir,
        tables_dir=run_dir,
        config_snapshot_dir=run_dir / "config_snapshot",
        manifest_path=run_dir / "run_manifest.json",
    )


def _std_or_zero(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1))


def _figure2_rice_pixels_path() -> Path:
    data_dir = CONFIG.data.raw_weather.parents[1]
    return (
        data_dir
        / "artifacts"
        / "features"
        / "china_rice_calendar"
        / DEFAULT_REGIONAL_GRID_FIGURE_PERIOD
        / "middle_rice_pixels_0p05deg_median_min10_lat16_35.parquet"
    )


def _figure2_yangtze_river_path() -> Path:
    return CONFIG.data.china_boundary.parent / "yangtze_river.geojson"


def _load_figure2_rice_pixels(path: Path | str | None = None) -> pd.DataFrame:
    target_path = _figure2_rice_pixels_path() if path is None else Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing Figure 2 rice calendar pixels: {target_path}")

    pixels = pd.read_parquet(target_path).copy()
    missing = [column for column in ("lon", "lat") if column not in pixels.columns]
    if missing:
        raise ValueError(f"Figure 2 rice calendar pixels missing required columns: {missing}")
    pixels["lon"] = pd.to_numeric(pixels["lon"], errors="coerce")
    pixels["lat"] = pd.to_numeric(pixels["lat"], errors="coerce")
    pixels = pixels.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if pixels.empty:
        raise ValueError(f"Figure 2 rice calendar pixels contain no valid lon/lat rows: {target_path}")
    return pixels


def _load_figure2_yangtze_river(path: Path | str | None = None) -> gpd.GeoDataFrame:
    target_path = _figure2_yangtze_river_path() if path is None else Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing Figure 2 Yangtze River file: {target_path}")

    yangtze = gpd.read_file(target_path)
    if yangtze.empty:
        raise ValueError(f"Figure 2 Yangtze River file contains no features: {target_path}")
    if yangtze.crs is None:
        yangtze = yangtze.set_crs("EPSG:4326")
    else:
        yangtze = yangtze.to_crs("EPSG:4326")
    return yangtze


def _figure2_weather_climatology_from_daily(
    weather: pd.DataFrame,
    *,
    months: tuple[int, ...] = FIGURE2_CLIMATE_MONTHS,
    year_range: tuple[int, int] = FIGURE2_WEATHER_YEAR_RANGE,
) -> pd.DataFrame:
    required = ("SID", "Date", "year", "TemAver", "Precipitation")
    missing = [column for column in required if column not in weather.columns]
    if missing:
        raise ValueError(f"Figure 2 weather climatology missing required columns: {missing}")

    frame = weather.loc[:, list(required)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ("SID", "year", "TemAver", "Precipitation"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["SID", "Date", "year", "TemAver", "Precipitation"]).copy()
    frame["year"] = frame["year"].astype(int)
    frame["month"] = frame["Date"].dt.month
    start_year, end_year = year_range
    frame = frame[
        frame["month"].isin(months)
        & frame["year"].between(int(start_year), int(end_year))
    ].copy()
    if frame.empty:
        raise ValueError(
            f"Figure 2 weather climatology has no rows for months {months} "
            f"and years {start_year}-{end_year}."
        )

    site_year_month = (
        frame.groupby(["SID", "year", "month"], as_index=False)
        .agg(
            temperature=("TemAver", "mean"),
            precipitation=("Precipitation", "sum"),
        )
    )
    site_month = (
        site_year_month.groupby(["SID", "month"], as_index=False)
        .agg(
            temperature=("temperature", "mean"),
            precipitation=("precipitation", "mean"),
        )
    )
    monthly = (
        site_month.groupby("month")
        .agg(
            temperature_mean=("temperature", "mean"),
            temperature_std=("temperature", _std_or_zero),
            precipitation_mean=("precipitation", "mean"),
            precipitation_std=("precipitation", _std_or_zero),
            n_sites=("SID", "nunique"),
        )
        .reindex(months)
        .reset_index()
        .rename(columns={"index": "month"})
    )
    monthly["month_label"] = pd.to_datetime(
        {
            "year": [FIGURE2_DAYLENGTH_YEAR] * len(monthly),
            "month": monthly["month"],
            "day": [1] * len(monthly),
        }
    ).dt.strftime("%b")
    monthly[["temperature_std", "precipitation_std"]] = monthly[
        ["temperature_std", "precipitation_std"]
    ].fillna(0.0)
    return monthly


def _load_figure2_weather_climatology(path: Path | str | None = None) -> pd.DataFrame:
    weather = load_raw_weather(Path(path) if path is not None else None)
    return _figure2_weather_climatology_from_daily(weather)


def _figure2_map_dates_to_fixed_year(
    dates: pd.Series | list[object] | np.ndarray,
    *,
    year: int = FIGURE2_DAYLENGTH_YEAR,
) -> pd.Series:
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    mapped = []
    for value in parsed:
        if pd.isna(value):
            mapped.append(pd.NaT)
            continue
        try:
            mapped.append(pd.Timestamp(year=int(year), month=int(value.month), day=int(value.day)))
        except ValueError:
            mapped.append(pd.Timestamp(year=int(year), month=2, day=28))
    return pd.Series(mapped, index=parsed.index, dtype="datetime64[ns]")


def _figure2_phenology_calendar_from_raw(phenology: pd.DataFrame) -> pd.DataFrame:
    missing = [column for _, column in FIGURE2_PHENOLOGY_STAGES if column not in phenology.columns]
    if missing:
        raise ValueError(f"Figure 2 phenology calendar missing required columns: {missing}")

    sid_column = "station ID" if "station ID" in phenology.columns else "SID" if "SID" in phenology.columns else None
    frames = []
    for stage_index, (stage_label, column) in enumerate(FIGURE2_PHENOLOGY_STAGES):
        original_dates = pd.to_datetime(phenology[column], errors="coerce")
        frame = pd.DataFrame(
            {
                "stage": stage_label,
                "stage_index": stage_index,
                "stage_date": original_dates,
                "plot_date": _figure2_map_dates_to_fixed_year(original_dates),
            }
        )
        if sid_column is not None:
            frame["SID"] = pd.to_numeric(phenology[sid_column], errors="coerce")
        if "year" in phenology.columns:
            frame["year"] = pd.to_numeric(phenology["year"], errors="coerce")
        frames.append(frame.dropna(subset=["plot_date"]))

    calendar = pd.concat(frames, ignore_index=True)
    if calendar.empty:
        raise ValueError("Figure 2 phenology calendar has no valid stage dates.")
    calendar["plot_date_num"] = mdates.date2num(calendar["plot_date"].to_numpy(dtype="datetime64[ms]"))
    return calendar.sort_values(["stage_index", "plot_date"]).reset_index(drop=True)


def _load_figure2_phenology_calendar(path: Path | str | None = None) -> pd.DataFrame:
    target_path = CONFIG.data.raw_phenology if path is None else Path(path)
    phenology = pd.read_excel(target_path)
    return _figure2_phenology_calendar_from_raw(phenology)


def _load_figure2_station_latitudes(path: Path | str | None = None) -> pd.Series:
    target_path = CONFIG.data.raw_phenology if path is None else Path(path)
    catalog = pd.read_excel(target_path)
    if "lat" not in catalog.columns:
        raise ValueError(f"Figure 2 station catalog missing required column: lat")
    catalog = catalog.copy()
    catalog["lat"] = pd.to_numeric(catalog["lat"], errors="coerce")
    sid_column = "station ID" if "station ID" in catalog.columns else "SID" if "SID" in catalog.columns else None
    columns = [sid_column, "lat"] if sid_column is not None else ["lat"]
    latitudes = catalog[columns].dropna(subset=["lat"]).drop_duplicates()["lat"]
    if latitudes.empty:
        raise ValueError(f"Figure 2 station catalog contains no valid latitude rows: {target_path}")
    return latitudes.reset_index(drop=True)


def _build_figure2_daylength_grid(
    latitudes: pd.Series | np.ndarray,
    *,
    n_latitudes: int = 120,
    year: int = FIGURE2_DAYLENGTH_YEAR,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    latitude_values = pd.to_numeric(pd.Series(latitudes), errors="coerce").dropna().to_numpy(dtype=float)
    if len(latitude_values) == 0:
        raise ValueError("Figure 2 daylength grid needs at least one valid station latitude.")

    lat_min = float(np.min(latitude_values))
    lat_max = float(np.max(latitude_values))
    if lat_min == lat_max:
        lat_min -= 0.5
        lat_max += 0.5
    lat_grid = np.linspace(lat_min, lat_max, int(n_latitudes))
    start_month = int(FIGURE2_DAYLENGTH_MONTHS[0])
    end_month = int(FIGURE2_DAYLENGTH_MONTHS[-1])
    end_day = pd.Period(f"{int(year)}-{end_month:02d}").days_in_month
    dates = pd.date_range(
        f"{int(year)}-{start_month:02d}-01",
        f"{int(year)}-{end_month:02d}-{end_day:02d}",
        freq="D",
    )
    calculator = DayLengthCalculator()
    daylength = np.array(
        [
            [calculator.day_length(date.year, date.month, date.day, float(latitude)) for date in dates]
            for latitude in lat_grid
        ],
        dtype=float,
    )
    return lat_grid, dates, daylength


def _load_figure2_daylength_grid(path: Path | str | None = None) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    latitudes = _load_figure2_station_latitudes(path)
    return _build_figure2_daylength_grid(latitudes)


def _plot_figure2_phenology_calendar(ax, calendar: pd.DataFrame) -> None:
    stage_labels = [stage_label for stage_label, _ in FIGURE2_PHENOLOGY_STAGES]
    positions = np.arange(1, len(stage_labels) + 1)
    stage_values = [
        calendar.loc[calendar["stage"] == stage_label, "plot_date_num"].to_numpy(dtype=float)
        for stage_label in stage_labels
    ]
    nonempty_values = [values for values in stage_values if len(values)]
    if not nonempty_values:
        raise ValueError("Figure 2 phenology calendar has no values to plot.")

    box = ax.boxplot(
        stage_values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "#333333", "linewidth": 0.8},
    )
    stage_colors = FIGURE2_PHENOLOGY_STAGE_COLORS
    for patch, color in zip(box["boxes"], stage_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for artist_group in ("boxes", "medians", "whiskers", "caps"):
        for artist in box[artist_group]:
            artist.set_zorder(4)

    rng = np.random.default_rng(20260509)
    for position, values, color in zip(positions, stage_values, stage_colors, strict=False):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            s=7,
            color=color,
            alpha=0.16,
            linewidths=0.0,
            rasterized=True,
            zorder=2,
        )

    year = FIGURE2_DAYLENGTH_YEAR
    y_min = mdates.date2num(pd.Timestamp(year, 3, 1))
    y_max = mdates.date2num(pd.Timestamp(year, 11, 30))
    month_ticks = pd.date_range(f"{year}-03-01", f"{year}-11-01", freq="MS")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(mdates.date2num(month_ticks.to_pydatetime()))
    ax.set_yticklabels([f"{date:%b} {date.day}" for date in month_ticks])
    ax.set_xticks(positions)
    ax.set_xticklabels(stage_labels, rotation=35, ha="right")
    ax.set_xlim(0.4, len(stage_labels) + 0.6)
    # ax.set_xlabel("Phenological stage")
    ax.set_ylabel("Calendar date")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)


def add_north(ax, labelsize=10, loc_x=0.92, loc_y=0.92, width=0.04, height=0.13, pad=0.14):
    """
    Add a north arrow to a map.

    Parameters:
    ax : matplotlib.axes.Axes
        The axes to which the north arrow will be added.
    labelsize : int, optional
        The font size of the 'N' label. Default is 10.
    loc_x : float, optional
        The x-location of the arrow's base as a fraction of the axes width. Default is 0.92.
    loc_y : float, optional
        The y-location of the arrow's base as a fraction of the axes height. Default is 0.92.
    width : float, optional
        The width of the arrow as a fraction of the axes width. Default is 0.04.
    height : float, optional
        The height of the arrow as a fraction of the axes height. Default is 0.13.
    pad : float, optional
        The padding between the arrow and the 'N' label as a fraction of the axes height. Default is 0.14.

    Returns:
    None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx

    # Calculate the positions for the north arrow
    left = [minx + xlen * (loc_x - width * 0.5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * 0.5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * 0.4]

    # Create the north arrow as a polygon
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.add_patch(triangle)

    # Add the 'N' label
    ax.text(s='N',
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    


def _add_scale_bar(ax, length_km: float = 500.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_range = float(xmax - xmin)
    y_range = float(ymax - ymin)
    if x_range <= 0.0 or y_range <= 0.0:
        return

    y = ymin + y_range * 0.075
    km_per_degree_lon = 111.32 * np.cos(np.deg2rad(y))
    if km_per_degree_lon <= 0.0:
        return

    length_degrees = float(length_km) / km_per_degree_lon
    x_end = xmax - x_range * 0.06
    x_start = x_end - length_degrees
    tick_height = y_range * 0.014
    color = "#222222"
    ax.plot(
        [x_start, x_end],
        [y, y],
        color=color,
        linewidth=1.25,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.plot([x_start, x_start], [y - tick_height, y + tick_height], color=color, linewidth=1.0, zorder=6)
    ax.plot([x_end, x_end], [y - tick_height, y + tick_height], color=color, linewidth=1.0, zorder=6)
    ax.text(
        (x_start + x_end) / 2.0,
        y + y_range * 0.03,
        f"{int(round(length_km))} km",
        ha="center",
        va="bottom",
        fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        zorder=6,
    )


def _add_map_orientation(ax) -> None:
    add_north(ax, labelsize=MANUSCRIPT_FONT_SIZES["annotation"], loc_x=0.93, loc_y=0.94, width=0.03, height=0.1, pad=0.14)
    _add_scale_bar(ax)


def _load_seed_metric_aggregates(run_dir_or_paths: Path | RunPaths) -> pd.DataFrame:
    run_paths = _coerce_run_paths(run_dir_or_paths)
    model_order = _comparison_models()
    frames = []
    for seed_dir in resolve_seed_eval_dirs(run_paths):
        for task in MAIN_DVR_TASKS:
            for model in model_order:
                path = seed_dir / f"{task}_{model}_metrics.csv"
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                frame = frame[frame["stage"].isin(SUMMARY_STAGE_ORDER)].copy()
                if frame.empty:
                    continue
                frame["seed_dir"] = seed_dir.name
                frame["task"] = task
                frame["model"] = model
                frames.append(frame[["task", "stage", "model", "mae", "rmse", "bias", "r2", "seed_dir"]])
    if not frames:
        return pd.DataFrame(columns=["task", "stage"])

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["task", "stage", "model"], as_index=False, observed=True)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", _std_or_zero),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", _std_or_zero),
            bias_mean=("bias", "mean"),
            bias_std=("bias", _std_or_zero),
            r2_mean=("r2", "mean"),
            r2_std=("r2", _std_or_zero),
        )
    )

    rows: list[dict[str, float | str]] = []
    for (task, stage), subset in grouped.groupby(["task", "stage"], observed=True, sort=True):
        row: dict[str, float | str] = {"task": str(task), "stage": str(stage)}
        for _, metric_row in subset.iterrows():
            model_name = str(metric_row["model"])
            for metric in SUMMARY_METRICS:
                row[f"{model_name}_{metric}_mean"] = float(metric_row[f"{metric}_mean"])
                row[f"{model_name}_{metric}_std"] = float(metric_row[f"{metric}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_aggregated_dvr_summary(run_dir: Path) -> pd.DataFrame:
    path = Path(run_dir) / "dvr_relative_change_summary.csv"
    summary = pd.read_csv(path)
    summary = summary[summary["task"].isin(MAIN_DVR_TASKS)].copy()
    model_order = _comparison_models()
    seed_metric_summary: pd.DataFrame | None = None
    indexed_seed_metric_summary: pd.DataFrame | None = None
    summary = summary.set_index(["task", "stage"])
    for model in model_order:
        for metric in SUMMARY_METRICS:
            mean_name = f"{model}_{metric}_mean"
            raw_name = f"{model}_{metric}"
            std_name = f"{model}_{metric}_std"
            if mean_name not in summary.columns:
                if raw_name in summary.columns:
                    summary[mean_name] = summary[raw_name]
                else:
                    if seed_metric_summary is None:
                        seed_metric_summary = _load_seed_metric_aggregates(run_dir)
                        indexed_seed_metric_summary = seed_metric_summary.set_index(["task", "stage"]) if not seed_metric_summary.empty else pd.DataFrame()
                    if indexed_seed_metric_summary is None or indexed_seed_metric_summary.empty or mean_name not in indexed_seed_metric_summary.columns:
                        raise ValueError(f"Missing required aggregated summary column: {mean_name}")
                    summary[mean_name] = indexed_seed_metric_summary[mean_name]
            if std_name not in summary.columns:
                if raw_name in summary.columns:
                    summary[std_name] = 0.0
                elif mean_name in summary.columns:
                    summary[std_name] = 0.0
                else:
                    if indexed_seed_metric_summary is None or indexed_seed_metric_summary.empty or std_name not in indexed_seed_metric_summary.columns:
                        raise ValueError(f"Missing required aggregated summary column: {std_name}")
                    summary[std_name] = indexed_seed_metric_summary[std_name]
    summary = summary.reset_index()
    summary["task"] = pd.Categorical(summary["task"], categories=MAIN_DVR_TASKS, ordered=True)
    summary["stage"] = pd.Categorical(summary["stage"], categories=SUMMARY_STAGE_ORDER, ordered=True)
    return summary.sort_values(["task", "stage"]).reset_index(drop=True)


def _load_tidy_aggregated_dvr_summary(run_dir: Path) -> pd.DataFrame:
    summary = _load_aggregated_dvr_summary(run_dir)
    model_order = _comparison_models()
    frames = []
    for model in model_order:
        for metric in SUMMARY_METRICS:
            frame = summary[["task", "stage", f"{model}_{metric}_mean", f"{model}_{metric}_std"]].copy()
            frame["model"] = model
            frame["metric"] = metric
            frame = frame.rename(
                columns={
                    f"{model}_{metric}_mean": "mean",
                    f"{model}_{metric}_std": "std",
                }
            )
            frames.append(frame)
    tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["task", "stage", "model", "metric", "mean", "std"])
    tidy["task"] = pd.Categorical(tidy["task"], categories=MAIN_DVR_TASKS, ordered=True)
    tidy["stage"] = pd.Categorical(tidy["stage"], categories=SUMMARY_STAGE_ORDER, ordered=True)
    tidy["model"] = pd.Categorical(tidy["model"], categories=model_order, ordered=True)
    tidy["metric"] = pd.Categorical(tidy["metric"], categories=SUMMARY_METRICS, ordered=True)
    return tidy.sort_values(["metric", "task", "stage", "model"]).reset_index(drop=True)


def _load_fold_aggregated_split_metric_summary(run_dir_or_paths: Path | RunPaths, *, absolute_bias: bool = False) -> pd.DataFrame:
    run_paths = _coerce_run_paths(run_dir_or_paths)
    model_order = _comparison_models()
    required_columns = ("task", "fold", "model", "stage", "mae", "rmse", "bias", "r2")
    frames = []
    missing_files: list[Path] = []

    for seed_dir in resolve_seed_eval_dirs(run_paths):
        for task in MAIN_DVR_TASKS:
            for model in model_order:
                path = seed_dir / f"{task}_{model}_split_metrics.csv"
                if not path.exists():
                    missing_files.append(path)
                    continue
                frame = pd.read_csv(path)
                missing_columns = [column for column in required_columns if column not in frame.columns]
                if missing_columns:
                    missing_display = ", ".join(missing_columns)
                    raise ValueError(f"{path.name} is missing required columns: {missing_display}")
                frame = frame[list(required_columns)].copy()
                frame = frame[frame["stage"].isin(SUMMARY_STAGE_ORDER)].copy()
                if frame.empty:
                    raise ValueError(f"{path.name} does not contain any required summary stages")
                frame["seed_dir"] = seed_dir.name
                frame["task"] = task
                frame["model"] = model
                frames.append(frame)

    if missing_files:
        missing_display = "\n".join(str(path) for path in missing_files[:8])
        remaining = len(missing_files) - min(len(missing_files), 8)
        suffix = "" if remaining <= 0 else f"\n... and {remaining} more"
        raise ValueError(f"Missing required split-metric files for Figures 5/6/7:\n{missing_display}{suffix}")

    if not frames:
        raise ValueError("No split metrics were found for Figures 5/6/7 aggregation")

    combined = pd.concat(frames, ignore_index=True)
    combined["task"] = pd.Categorical(combined["task"], categories=MAIN_DVR_TASKS, ordered=True)
    combined["stage"] = pd.Categorical(combined["stage"], categories=SUMMARY_STAGE_ORDER, ordered=True)
    combined["model"] = pd.Categorical(combined["model"], categories=model_order, ordered=True)
    per_fold = (
        combined.groupby(["task", "stage", "model", "fold"], as_index=False, observed=True)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            bias=("bias", "mean"),
            r2=("r2", "mean"),
        )
    )
    summary_source = per_fold.copy()
    if absolute_bias:
        summary_source["bias"] = summary_source["bias"].abs()
    grouped = (
        summary_source.groupby(["task", "stage", "model"], as_index=False, observed=True)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", _std_or_zero),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", _std_or_zero),
            bias_mean=("bias", "mean"),
            bias_std=("bias", _std_or_zero),
            r2_mean=("r2", "mean"),
            r2_std=("r2", _std_or_zero),
        )
    )

    frames = []
    for metric in SUMMARY_METRICS:
        frame = grouped[["task", "stage", "model", f"{metric}_mean", f"{metric}_std"]].copy()
        frame["metric"] = metric
        frame = frame.rename(columns={f"{metric}_mean": "mean", f"{metric}_std": "std"})
        frames.append(frame)
    tidy = pd.concat(frames, ignore_index=True)
    tidy["task"] = pd.Categorical(tidy["task"], categories=MAIN_DVR_TASKS, ordered=True)
    tidy["stage"] = pd.Categorical(tidy["stage"], categories=SUMMARY_STAGE_ORDER, ordered=True)
    tidy["model"] = pd.Categorical(tidy["model"], categories=model_order, ordered=True)
    tidy["metric"] = pd.Categorical(tidy["metric"], categories=SUMMARY_METRICS, ordered=True)
    return tidy.sort_values(["metric", "task", "stage", "model"]).reset_index(drop=True)


def _prediction_wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    long_rows: list[pd.DataFrame] = []
    base_columns = ["SID", "year", "task", "model"]
    if "fold" in frame.columns:
        base_columns.append("fold")
    for stage in STAGE_ORDER:
        long_rows.append(
            frame.loc[:, base_columns].assign(
                stage=stage,
                obs_doy=pd.to_numeric(frame[f"obs_{stage}"], errors="coerce"),
                pred_doy=pd.to_numeric(frame[f"pred_{stage}"], errors="coerce"),
            )
        )
    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    long_df["bias"] = long_df["pred_doy"] - long_df["obs_doy"]
    return long_df


def _load_seed_prediction_long(run_dir_or_paths: Path | RunPaths) -> pd.DataFrame:
    run_paths = _coerce_run_paths(run_dir_or_paths)
    model_order = _comparison_models()
    frames = []
    for seed_dir in resolve_seed_eval_dirs(run_paths):
        for task in MAIN_DVR_TASKS:
            for model in model_order:
                path = seed_dir / f"{task}_{model}_predictions.csv"
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                if "label" in frame.columns:
                    frame = frame[frame["label"] == "test"].copy()
                if frame.empty:
                    continue
                long_df = _prediction_wide_to_long(frame)
                long_df["seed_dir"] = seed_dir.name
                frames.append(long_df)
    if not frames:
        return pd.DataFrame(columns=["SID", "year", "task", "model", "fold", "stage", "obs_doy", "pred_doy", "bias", "seed_dir"])

    combined = pd.concat(frames, ignore_index=True)
    combined["task"] = pd.Categorical(combined["task"], categories=MAIN_DVR_TASKS, ordered=True)
    combined["model"] = pd.Categorical(combined["model"], categories=model_order, ordered=True)
    combined["stage"] = pd.Categorical(combined["stage"], categories=STAGE_ORDER, ordered=True)
    sort_cols = ["task", "model", "stage", "seed_dir"]
    if "fold" in combined.columns:
        sort_cols.append("fold")
    sort_cols.extend(["SID", "year"])
    return combined.sort_values(sort_cols).reset_index(drop=True)


def _load_seed_averaged_prediction_long(run_dir_or_paths: Path | RunPaths) -> pd.DataFrame:
    combined = _load_seed_prediction_long(run_dir_or_paths)
    model_order = _comparison_models()
    if combined.empty:
        return pd.DataFrame(columns=["SID", "year", "task", "model", "fold", "stage", "obs_doy", "pred_doy", "bias", "n_seeds"])

    group_cols = ["SID", "year", "task", "model", "stage", "obs_doy"]
    if "fold" in combined.columns:
        group_cols.insert(4, "fold")
    averaged = (
        combined.groupby(group_cols, as_index=False, observed=True)
        .agg(pred_doy=("pred_doy", "mean"), n_seeds=("seed_dir", "nunique"))
    )
    averaged["bias"] = averaged["pred_doy"] - averaged["obs_doy"]
    averaged["task"] = pd.Categorical(averaged["task"], categories=MAIN_DVR_TASKS, ordered=True)
    averaged["model"] = pd.Categorical(averaged["model"], categories=model_order, ordered=True)
    averaged["stage"] = pd.Categorical(averaged["stage"], categories=STAGE_ORDER, ordered=True)
    sort_cols = ["task", "model", "stage"]
    if "fold" in averaged.columns:
        sort_cols.append("fold")
    sort_cols.extend(["SID", "year"])
    return averaged.sort_values(sort_cols).reset_index(drop=True)


def _select_representative_seed(subset: pd.DataFrame) -> str:
    if subset.empty:
        raise ValueError("Representative seed selection requires at least one row.")
    if "seed_dir" not in subset.columns:
        raise ValueError("Representative seed selection requires a seed_dir column.")

    per_seed_rows: list[dict[str, float | str]] = []
    for seed_dir, seed_frame in subset.groupby("seed_dir", sort=True):
        per_seed_rows.append({"seed_dir": str(seed_dir), "mae": _mae(seed_frame["obs_doy"], seed_frame["pred_doy"])})
    per_seed = pd.DataFrame(per_seed_rows)
    target_mae = float(per_seed["mae"].mean())
    per_seed["distance_to_mean"] = (per_seed["mae"] - target_mae).abs()
    return str(per_seed.sort_values(["distance_to_mean", "seed_dir"]).iloc[0]["seed_dir"])


def _select_global_representative_seed(subset: pd.DataFrame) -> str:
    if subset.empty:
        raise ValueError("Global representative seed selection requires at least one row.")
    required_columns = {"task", "model", "seed_dir", "obs_doy", "pred_doy"}
    missing = required_columns - set(subset.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Global representative seed selection requires columns: {missing_display}")

    panel_rows: list[dict[str, float | str]] = []
    for (task, model, seed_dir), seed_frame in subset.groupby(["task", "model", "seed_dir"], sort=True, observed=True):
        panel_rows.append(
            {
                "task": str(task),
                "model": str(model),
                "seed_dir": str(seed_dir),
                "mae": _mae(seed_frame["obs_doy"], seed_frame["pred_doy"]),
            }
        )
    panel_metrics = pd.DataFrame(panel_rows)
    panel_metrics["panel_mean_mae"] = panel_metrics.groupby(["task", "model"], observed=True)["mae"].transform("mean")
    panel_metrics["distance_to_panel_mean"] = (panel_metrics["mae"] - panel_metrics["panel_mean_mae"]).abs()
    seed_scores = (
        panel_metrics.groupby("seed_dir", observed=True, as_index=False)
        .agg(total_abs_dev=("distance_to_panel_mean", "sum"))
        .sort_values(["total_abs_dev", "seed_dir"])
        .reset_index(drop=True)
    )
    return str(seed_scores.iloc[0]["seed_dir"])


def _load_representative_seed_prediction_long(run_dir_or_paths: Path | RunPaths, *, stage: str = "maturity") -> pd.DataFrame:
    combined = _load_seed_prediction_long(run_dir_or_paths)
    if combined.empty:
        return combined

    stage_subset = combined[combined["stage"] == stage].copy()
    selections: list[pd.DataFrame] = []
    model_order = _comparison_models()
    for task in MAIN_DVR_TASKS:
        for model in model_order:
            subset = stage_subset[(stage_subset["task"] == task) & (stage_subset["model"] == model)]
            if subset.empty:
                continue
            representative_seed = _select_representative_seed(subset)
            selections.append(subset[subset["seed_dir"] == representative_seed].copy())
    if not selections:
        return pd.DataFrame(columns=stage_subset.columns)

    selected = pd.concat(selections, ignore_index=True)
    sort_cols = ["task", "model", "seed_dir"]
    if "fold" in selected.columns:
        sort_cols.append("fold")
    sort_cols.extend(["SID", "year"])
    return selected.sort_values(sort_cols).reset_index(drop=True)


def _load_global_representative_seed_prediction_long(run_dir_or_paths: Path | RunPaths, *, stage: str = "maturity") -> pd.DataFrame:
    combined = _load_seed_prediction_long(run_dir_or_paths)
    if combined.empty:
        return combined

    stage_subset = combined[combined["stage"] == stage].copy()
    if stage_subset.empty:
        return stage_subset
    representative_seed = _select_global_representative_seed(stage_subset)
    selected = stage_subset[stage_subset["seed_dir"] == representative_seed].copy()
    sort_cols = ["task", "model", "seed_dir"]
    if "fold" in selected.columns:
        sort_cols.append("fold")
    sort_cols.extend(["SID", "year"])
    return selected.sort_values(sort_cols).reset_index(drop=True)


def _mae(obs: pd.Series, pred: pd.Series) -> float:
    return float((pred - obs).abs().mean())


def _rmse(obs: pd.Series, pred: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(pred - obs))))


def _r2(obs: pd.Series, pred: pd.Series) -> float:
    obs_values = pd.to_numeric(obs, errors="coerce")
    pred_values = pd.to_numeric(pred, errors="coerce")
    mask = obs_values.notna() & pred_values.notna()
    if mask.sum() < 2:
        return float("nan")
    residual = np.square(pred_values[mask] - obs_values[mask]).sum()
    total = np.square(obs_values[mask] - obs_values[mask].mean()).sum()
    if total <= 1e-12:
        return float("nan")
    return float(1.0 - residual / total)


def _ci95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _fold_color(fold: object) -> tuple[float, float, float] | tuple[float, float, float, float]:
    if pd.isna(fold):
        return FOLD_COLOR_PALETTE[0]
    try:
        idx = (int(fold) - 1) % len(FOLD_COLOR_PALETTE)
    except (TypeError, ValueError):
        idx = 0
    return FOLD_COLOR_PALETTE[idx]


def framework(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=MANUSCRIPT_FIGURE_SIZES["framework"])
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    route_steps = [
        ("Data\nfoundation", "Phenology records\nDaily weather\nSite-year metadata"),
        ("Stage-level\nrepresentation", "Stage weather windows\nStage requirements\nDVR-ready inputs"),
        ("Four-model\ncomparison", ""),
        ("Extrapolation\nvalidation", "sample | site | year\nProtocol-aligned\nrollout"),
        ("Mechanistic\ndiagnosis", "Stage errors\nBias propagation\nCorrection response"),
        ("Regional\ndeployment", "Grid rollout\nRemote-sensing\ncomparison"),
    ]
    fill_colors = [
        "#edf6f3",
        "#f5f2e8",
        "#eef1fa",
        "#fff3e8",
        "#f4edf8",
        "#eef4f8",
    ]

    xs = np.linspace(0.08, 0.92, len(route_steps))
    box_width = 0.13
    box_height = 0.42
    y0 = 0.42
    for idx, (x, (title, details)) in enumerate(zip(xs, route_steps, strict=False)):
        box = FancyBboxPatch(
            (x - box_width / 2, y0),
            box_width,
            box_height,
            boxstyle="round,pad=0.016,rounding_size=0.018",
            fc=fill_colors[idx],
            ec=FIGURE_STYLE.accent,
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y0 + box_height - 0.075,
            title,
            ha="center",
            va="center",
            fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
            fontweight="bold",
            color="#222222",
            linespacing=1.05,
        )
        if details:
            ax.text(
                x,
                y0 + 0.16,
                details,
                ha="center",
                va="center",
                fontsize=MANUSCRIPT_FONT_SIZES["annotation"] - 0.7,
                color="#333333",
                linespacing=1.18,
            )
        else:
            pill_specs = [
                ("m0_t", -0.032, 0.215),
                ("m0_dvr", 0.032, 0.215),
                ("m1_v2_dvr", -0.032, 0.135),
                ("m1_dvr_con", 0.032, 0.135),
            ]
            for model, dx, dy in pill_specs:
                pill = FancyBboxPatch(
                    (x + dx - 0.030, y0 + dy - 0.025),
                    0.060,
                    0.050,
                    boxstyle="round,pad=0.006,rounding_size=0.012",
                    fc=MODEL_COLORS[model],
                    ec=MODEL_COLORS[model],
                    alpha=0.28,
                    linewidth=0.8,
                )
                ax.add_patch(pill)
                ax.text(
                    x + dx,
                    y0 + dy,
                    MODEL_DISPLAY[model],
                    ha="center",
                    va="center",
                    fontsize=MANUSCRIPT_FONT_SIZES["annotation"] - 1.8,
                    fontweight="bold",
                    color="#222222",
                )
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - box_width / 2 - 0.008, y0 + box_height / 2),
            xytext=(xs[i] + box_width / 2 + 0.008, y0 + box_height / 2),
            arrowprops={"arrowstyle": "->", "lw": 1.05, "color": FIGURE_STYLE.accent},
        )

    output_band = FancyBboxPatch(
        (0.07, 0.13),
        0.86,
        0.14,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        fc="#f7f8f5",
        ec="#c9d4d1",
        linewidth=0.9,
    )
    ax.add_patch(output_band)
    ax.text(
        0.5,
        0.20,
        "Daily DVR -> stage accumulation -> rollout dates -> validation evidence -> spatial application",
        ha="center",
        va="center",
        fontsize=MANUSCRIPT_FONT_SIZES["label"],
        color="#222222",
    )
    plt.tight_layout()
    return _save(fig, "framework.png", output_dir)


def study_area(output_dir: Path) -> Path:
    china, provinces, sites, _ = _load_study_area_inputs()
    rice_pixels = _load_figure2_rice_pixels()
    yangtze_river = _load_figure2_yangtze_river()
    phenology_calendar = _load_figure2_phenology_calendar()
    climatology = _load_figure2_weather_climatology()
    lat_grid, daylength_dates, daylength = _load_figure2_daylength_grid()

    fig = plt.figure(
        figsize=MANUSCRIPT_FIGURE_SIZES["study_domain"],
        constrained_layout=True,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.04)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.2, 1.0, 1.0],
        width_ratios=[1.1, 1.0],
        wspace=-1.5,
    )
    ax_map = fig.add_subplot(grid[0, 0])
    ax_pheno = fig.add_subplot(grid[0, 1])
    ax_climate = fig.add_subplot(grid[1, :])
    ax_daylength = fig.add_subplot(grid[2, :])

    rice_background = ax_map.scatter(
        rice_pixels["lon"],
        rice_pixels["lat"],
        s=5,
        marker="s",
        color="#5B7438",
        alpha=0.6,
        linewidths=0.0,
        rasterized=True,
        label="Southern middle-rice",
        zorder=1,
    )
    china.boundary.plot(ax=ax_map, color="black", linewidth=0.6, zorder=2)
    provinces.boundary.plot(ax=ax_map, color="#999999", linewidth=0.3, zorder=2)
    yangtze_color = "#2F6FA5"
    yangtze_river.plot(
        ax=ax_map,
        color=yangtze_color,
        linewidth=1.05,
        alpha=0.95,
        label="Yangtze River",
        zorder=3,
    )
    yangtze_handle = Line2D([0], [0], color=yangtze_color, linewidth=1.05, label="Yangtze River")
    site_lon = pd.to_numeric(sites["lon"], errors="coerce")
    site_lat = pd.to_numeric(sites["lat"], errors="coerce")
    site_mask = site_lon.notna() & site_lat.notna()
    site_points = ax_map.scatter(
        site_lon.loc[site_mask],
        site_lat.loc[site_mask],
        s=22,
        color=FIGURE_STYLE.secondary,
        edgecolors="white",
        linewidths=0.4,
        label="Observation sites",
        zorder=4,
    )
    xmin, xmax, ymin, ymax = CONFIG.figures.map_extent
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_anchor("E")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.legend(
        handles=[yangtze_handle, rice_background, site_points],
        loc="lower left",
        frameon=False,
        fontsize=MANUSCRIPT_FONT_SIZES["legend"],
    )
    add_north(ax_map, labelsize=MANUSCRIPT_FONT_SIZES["annotation"], loc_x=0.92, loc_y=0.92, width=0.04, height=0.13, pad=0.14)
    _add_scale_bar(ax_map)
    _plot_figure2_phenology_calendar(ax_pheno, phenology_calendar)
    ax_pheno.yaxis.tick_right()
    ax_pheno.yaxis.set_label_position("right")

    months = climatology["month"].to_numpy(dtype=float)
    month_labels = climatology["month_label"].tolist()
    temperature_mean = climatology["temperature_mean"].to_numpy(dtype=float)
    temperature_std = climatology["temperature_std"].to_numpy(dtype=float)
    precipitation_mean = climatology["precipitation_mean"].to_numpy(dtype=float)
    precipitation_std = climatology["precipitation_std"].to_numpy(dtype=float)
    ax_precip = ax_climate.twinx()
    precip_bars = ax_precip.bar(
        months,
        precipitation_mean,
        width=0.62,
        color="#6EA7D8",
        alpha=0.32,
        edgecolor="none",
        label="Precipitation",
        zorder=1,
    )
    ax_precip.errorbar(
        months,
        precipitation_mean,
        yerr=precipitation_std,
        fmt="none",
        ecolor="#34699A",
        elinewidth=0.75,
        capsize=2,
        alpha=0.85,
        zorder=2,
    )
    ax_climate.set_zorder(ax_precip.get_zorder() + 1)
    ax_climate.patch.set_visible(False)
    ax_climate.fill_between(
        months,
        temperature_mean - temperature_std,
        temperature_mean + temperature_std,
        color="#D95F4C",
        alpha=0.16,
        linewidth=0.0,
        zorder=3,
    )
    (temp_line,) = ax_climate.plot(
        months,
        temperature_mean,
        color="#D95F4C",
        marker="o",
        markersize=3.8,
        linewidth=1.4,
        label="Temperature",
        zorder=4,
    )
    ax_climate.set_ylabel("Mean temperature (°C)")
    ax_precip.set_ylabel("Monthly precipitation (mm)")
    ax_climate.set_xticks(months)
    ax_climate.set_xticklabels(month_labels)
    ax_climate.set_xlim(min(months) - 0.5, max(months) + 0.5)
    ax_climate.grid(axis="y", linestyle=":", alpha=0.35)
    for spine in ax_climate.spines.values():
        spine.set_visible(True)
    ax_precip.spines["right"].set_visible(True)
    ax_climate.legend(
        handles=[temp_line, precip_bars],
        labels=["Temperature", "Precipitation"],
        loc="upper right",
        frameon=False,
        fontsize=MANUSCRIPT_FONT_SIZES["legend"],
    )

    im = ax_daylength.imshow(
        daylength,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0, len(daylength_dates) - 1, float(lat_grid.min()), float(lat_grid.max())),
        cmap="viridis",
    )
    month_positions = []
    month_position_labels = []
    for month in FIGURE2_DAYLENGTH_MONTHS:
        positions = np.flatnonzero(daylength_dates.month == month)
        if len(positions) == 0:
            continue
        month_positions.append(float(positions.mean()))
        month_position_labels.append(pd.Timestamp(FIGURE2_DAYLENGTH_YEAR, month, 1).strftime("%b"))
    ax_daylength.set_xticks(month_positions)
    ax_daylength.set_xticklabels(month_position_labels)
    ax_daylength.set_xlabel("Month")
    ax_daylength.set_ylabel("Latitude (°N)")
    ax_daylength.spines["top"].set_visible(False)
    ax_daylength.spines["right"].set_visible(False)
    cbar = fig.colorbar(im, ax=ax_daylength, fraction=0.030, pad=-0.08)
    cbar.set_label("Daylength (h)", fontsize=MANUSCRIPT_FONT_SIZES["label"])
    cbar.ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"])

    for ax, label in (
        (ax_map, "a"),
        (ax_pheno, "b"),
        (ax_climate, "c"),
        (ax_daylength, "d"),
    ):
        _regional_add_panel_label(ax, label)

    return _save(fig, "study_area.png", output_dir)

def evaluation_protocols(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=MANUSCRIPT_FIGURE_SIZES["protocols"], sharey=True)
    panel_specs = [
        ("A. Sample random", "Seen sites years", "Seen sites years", "random samples from seen sites and years"),
        ("B. Site extrapolation", "Seen years", "Unseen sites", "unseen sites under seen years"),
        ("C. Year extrapolation", "Seen sites", "Unseen future years", "unseen years under seen sites"),
    ]
    colors = [FIGURE_STYLE.primary, FIGURE_STYLE.secondary, FIGURE_STYLE.accent]
    for ax, (title, train_label, test_label, subtitle), color in zip(axes, panel_specs, colors):
        ax.axis("off")
        train_box = FancyBboxPatch((0.08, 0.58), 0.34, 0.20, boxstyle="round,pad=0.03", fc="#eef6f3", ec=color, linewidth=1.2)
        test_box = FancyBboxPatch((0.58, 0.22), 0.34, 0.20, boxstyle="round,pad=0.03", fc="#fff3e8", ec=color, linewidth=1.2)
        ax.add_patch(train_box)
        ax.add_patch(test_box)
        ax.text(0.25, 0.68, "Train", ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["label"], fontweight="bold")
        ax.text(0.25, 0.61, train_label, ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["annotation"])
        ax.text(0.75, 0.32, "Test", ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["label"], fontweight="bold")
        ax.text(0.75, 0.25, test_label, ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["annotation"])
        ax.annotate(
            "",
            xy=(0.58, 0.42),
            xytext=(0.42, 0.58),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": color},
        )
        ax.text(0.5, 0.88, title, ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["label"], fontweight="bold")
        ax.text(0.5, 0.08, subtitle, ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["annotation"], color="#444444")
    return _save(fig, "evaluation_protocols.png", output_dir)


def overall_performance_summary(output_dir: Path, run_dir: Path) -> Path:
    tidy = _load_fold_aggregated_split_metric_summary(run_dir, absolute_bias=True)
    # tidy.to_csv(output_dir / "overall_performance_summary_data.csv", index=False)
    model_order = _comparison_models()
    summary = tidy[(tidy["stage"] == "all_stage") & (tidy["metric"].isin(["mae", "rmse", "bias", "r2"]))].copy()
    metric_rows = [("mae", "MAE (days)"), ("rmse", "RMSE (days)"), ("bias", "Absolute bias (days)"), ("r2", "R²")]
    fig, axes = plt.subplots(len(metric_rows), 1, figsize=MANUSCRIPT_FIGURE_SIZES["overall_performance"], sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(MAIN_DVR_TASKS))
    width = 0.70 / max(len(model_order), 1)
    center = (len(model_order) - 1) / 2.0
    legend_handles = []
    legend_labels = []

    for row_idx, (metric, y_label) in enumerate(metric_rows):
        ax = axes[row_idx]
        metric_frame = summary[summary["metric"] == metric]
        annotation_rows: list[tuple[float, float, float]] = []
        visible_values: list[float] = []
        ax.text(
            -0.055,
            1.03,
            f"({chr(ord('a') + row_idx)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=MANUSCRIPT_FONT_SIZES["panel"],
            fontweight="bold",
        )
        for model_idx, model in enumerate(model_order):
            model_frame = metric_frame[metric_frame["model"] == model].set_index("task").reindex(MAIN_DVR_TASKS)
            offsets = x + (model_idx - center) * width
            values = model_frame["mean"].to_numpy(dtype=float)
            yerr = model_frame["std"].to_numpy(dtype=float)
            bars = ax.bar(
                offsets,
                values,
                yerr=yerr,
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_DISPLAY[model],
                capsize=2.0,
                edgecolor="white",
                linewidth=0.35,
                error_kw={"elinewidth": 0.5, "capthick": 0.5},
            )
            if row_idx == 0:
                legend_handles.append(bars)
                legend_labels.append(MODEL_DISPLAY[model])
            for x_pos, value, err in zip(offsets, values, yerr, strict=False):
                if not np.isfinite(value):
                    continue
                err_value = float(err) if np.isfinite(err) else 0.0
                visible_values.append(float(value))
                annotation_rows.append((float(x_pos), float(value), float(value) + max(err_value, 0.0)))

        if annotation_rows:
            y_min = min(0.0, *visible_values)
            y_top = max(row[2] for row in annotation_rows)
            y_range = max(y_top - y_min, 0.2 if metric == "r2" else 1.0)
            text_offset = 0.035 * y_range
            for x_pos, value, y_base in annotation_rows:
                ax.text(
                    x_pos,
                    y_base + text_offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=MANUSCRIPT_FONT_SIZES["annotation"] - 1.5,
                    color="#333333",
                    clip_on=False,
                )
            ax.set_ylim(top=y_top + 0.18 * y_range)

        ax.set_ylabel(y_label)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.tick_params(axis="both", length=2.5, width=0.6)
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=False,
        ncol=len(model_order),
        loc="upper center",
        bbox_to_anchor=(0.53, 1.01),
        columnspacing=3.0,
        handlelength=3.4,
        handleheight=1.7,
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([TASK_DISPLAY[task] for task in MAIN_DVR_TASKS], rotation=0)
    plt.tight_layout()
    return _save(fig, "overall_performance_summary.png", output_dir)


def stage_specific_mae_decomposition(output_dir: Path, run_dir: Path) -> Path:
    tidy = _load_fold_aggregated_split_metric_summary(run_dir)
    model_order = _comparison_models()
    summary = tidy[(tidy["metric"] == "mae") & (tidy["stage"].isin(STAGE_ORDER))].copy()
    fig, axes = plt.subplots(3, 1, figsize=MANUSCRIPT_FIGURE_SIZES["three_row_bars"], sharex=True)
    x = np.arange(len(FIGURE6_STAGE_ORDER))
    width = 0.88 / max(len(model_order), 1)
    center = (len(model_order) - 1) / 2.0
    legend_handles = []
    legend_labels = []
    for row_idx, task in enumerate(MAIN_DVR_TASKS):
        ax = axes[row_idx]
        task_frame = summary[summary["task"] == task]
        annotation_rows: list[tuple[float, float, float]] = []
        visible_values: list[float] = []
        ax.text(
            -0.055,
            1.03,
            f"({chr(ord('a') + row_idx)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=MANUSCRIPT_FONT_SIZES["panel"],
            fontweight="bold",
        )
        for idx, model in enumerate(model_order):
            offsets = x + (idx - center) * width
            model_frame = task_frame[task_frame["model"] == model].set_index("stage").reindex(FIGURE6_STAGE_ORDER)
            values = model_frame["mean"].to_numpy(dtype=float)
            yerr = model_frame["std"].to_numpy(dtype=float)
            bars = ax.bar(
                offsets,
                values,
                yerr=yerr,
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_DISPLAY[model],
                capsize=2.0,
                edgecolor="white",
                linewidth=0.35,
                error_kw={"elinewidth": 0.5, "capthick": 0.5},
            )
            if row_idx == 0:
                legend_handles.append(bars)
                legend_labels.append(MODEL_DISPLAY[model])
            for x_pos, value, err in zip(offsets, values, yerr, strict=False):
                if not np.isfinite(value):
                    continue
                err_value = float(err) if np.isfinite(err) else 0.0
                visible_values.append(float(value))
                annotation_rows.append((float(x_pos), float(value), float(value) + max(err_value, 0.0)))
        if annotation_rows:
            y_min = min(0.0, *visible_values)
            y_top = max(row[2] for row in annotation_rows)
            y_range = max(y_top - y_min, 1.0)
            text_offset = 0.035 * y_range
            for x_pos, value, y_base in annotation_rows:
                ax.text(
                    x_pos,
                    y_base + text_offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=MANUSCRIPT_FONT_SIZES["annotation"] - 1.0,
                    color="#333333",
                    clip_on=False,
                )
            ax.set_ylim(top=y_top + 0.18 * y_range)
        ax.set_ylabel(f"{TASK_DISPLAY[task]}\nMAE (days)")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.tick_params(axis="both", length=2.5, width=0.6)
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=False,
        ncol=len(model_order),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=3.0,
        handlelength=3.4,
        handleheight=1.7,
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([STAGE_DISPLAY[stage] for stage in FIGURE6_STAGE_ORDER], ha="center", rotation=0)
    plt.tight_layout()
    return _save(fig, "stage_specific_mae_decomposition.png", output_dir)


def model_design_ladder(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=MANUSCRIPT_FIGURE_SIZES["figure_4"])
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    block_width = 0.17
    block_height = 0.23
    model_blocks = [
        ("m0_t", "Temperature contribution\n+ stage requirement", 0.05, 0.64),
        ("m0_dvr", "Temperature x photoperiod\n+ stage requirement", 0.29, 0.58),
        ("m1_v2_dvr", "Photothermal DVR\nx daily correction", 0.53, 0.52),
        ("m1_dvr_con", "DRC\n+ constrained\ncontext state", 0.77, 0.46),
    ]
    progression_labels = [
        "Add photoperiod",
        "Learn daily correction",
        "Constrain background use",
    ]

    for model, description, x0, y0 in model_blocks:
        color = MODEL_COLORS[model]
        block = FancyBboxPatch(
            (x0, y0),
            block_width,
            block_height,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            fc=color,
            ec=color,
            alpha=0.18,
            linewidth=1.4,
        )
        ax.add_patch(block)
        ax.text(
            x0 + block_width / 2,
            y0 + block_height - 0.055,
            MODEL_DISPLAY[model],
            ha="center",
            va="center",
            fontsize=MANUSCRIPT_FONT_SIZES["panel"],
            fontweight="bold",
            color=color,
        )
        ax.text(
            x0 + block_width / 2,
            y0 + block_height / 2 - 0.035,
            description,
            ha="center",
            va="center",
            fontsize=MANUSCRIPT_FONT_SIZES["label"],
            color="#333333",
            linespacing=1.2,
        )

    for idx, label in enumerate(progression_labels):
        _model, _description, x0, y0 = model_blocks[idx]
        _next_model, _next_description, next_x0, next_y0 = model_blocks[idx + 1]
        start = (x0 + block_width + 0.02, y0 + block_height / 2)
        end = (next_x0 - 0.02, next_y0 + block_height / 2)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.25, "color": FIGURE_STYLE.accent},
        )
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.11,
            label,
            ha="center",
            va="center",
            fontsize=MANUSCRIPT_FONT_SIZES["annotation"] - 0.3,
            color=FIGURE_STYLE.accent,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.92},
        )

    output_band = FancyBboxPatch(
        (0.12, 0.15),
        0.76,
        0.15,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        fc="#f5f7f7",
        ec=FIGURE_STYLE.accent,
        linewidth=1.1,
    )
    ax.add_patch(output_band)
    ax.text(
        0.5,
        0.225,
        "Daily DVR -> stage accumulation -> rollout phenology dates",
        ha="center",
        va="center",
        fontsize=MANUSCRIPT_FONT_SIZES["label"],
        color="#222222",
        fontweight="bold",
    )
    for model, _description, x0, y0 in model_blocks:
        ax.annotate(
            "",
            xy=(x0 + block_width / 2, 0.31),
            xytext=(x0 + block_width / 2, y0 - 0.025),
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.9,
                "color": MODEL_COLORS[model],
                "alpha": 0.7,
            },
        )

    plt.tight_layout()
    return _save(fig, "model_design_ladder.png", output_dir)


def stage_bias_evolution(output_dir: Path, run_dir: Path) -> Path:
    summary = _load_fold_aggregated_split_metric_summary(run_dir)
    model_order = _comparison_models()
    summary = summary[(summary["metric"] == "bias") & (summary["stage"].isin(STAGE_ORDER))].copy()
    fig, axes = plt.subplots(3, 1, figsize=MANUSCRIPT_FIGURE_SIZES["figure_7"], sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(STAGE_ORDER))
    marker_cycle = ("o", "s", "^", "D", "P")
    legend_handles = []
    legend_labels = []
    for row_idx, task in enumerate(MAIN_DVR_TASKS):
        ax = axes[row_idx]
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.text(
            -0.055,
            1.03,
            f"({chr(ord('a') + row_idx)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=MANUSCRIPT_FONT_SIZES["panel"],
            fontweight="bold",
        )
        task_summary = summary[summary["task"] == task]
        for model_idx, model in enumerate(model_order):
            model_summary = task_summary[task_summary["model"] == model].set_index("stage").reindex(STAGE_ORDER)
            y = model_summary["mean"].to_numpy(dtype=float)
            line = ax.plot(
                x,
                y,
                marker=marker_cycle[model_idx % len(marker_cycle)],
                markersize=3.2,
                # markeredgecolor="white",
                markeredgewidth=0.55,
                linewidth=1.25,
                color=MODEL_COLORS[model],
                label=MODEL_DISPLAY[model],
            )[0]
            if row_idx == 0:
                legend_handles.append(line)
                legend_labels.append(MODEL_DISPLAY[model])
        ax.set_ylabel(f"{TASK_DISPLAY[task]}\nBias (days)")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.tick_params(axis="both", length=2.5, width=0.6)
        # for spine_name in ("top", "right"):
        #     ax.spines[spine_name].set_visible(False)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=False,
        ncol=len(model_order),
        loc="upper center",
        bbox_to_anchor=(0.53, 1.01),
        columnspacing=2.0,
        handlelength=3.4,
        handleheight=1.7,
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([STAGE_DISPLAY[stage] for stage in STAGE_ORDER])
    plt.tight_layout()
    return _save(fig, "stage_bias_evolution.png", output_dir)


def modifier_interpretability(
    output_dir: Path,
    run_dir: Path,
    *,
    stages: tuple[str, ...] | list[str] | None = None,
    stage: str | None = None,
) -> Path:
    if stages is not None and stage is not None:
        raise ValueError("Pass either 'stages' or 'stage', not both.")
    resolved_stages = tuple(stages) if stages is not None else ((stage,) if stage is not None else DEFAULT_FIGURE_8_STAGES)
    if len(resolved_stages) != 3:
        raise ValueError("Figure 8 uses a fixed 3 x 3 layout and requires exactly three modifier stages.")

    return _modifier_interpretability_stage_grid(
        output_dir,
        run_dir,
        stages=resolved_stages,
        stem="modifier_interpretability.png",
        figure_size=MANUSCRIPT_FIGURE_SIZES["figure_8"],
    )


def supplemental_modifier_interpretability_early_stages(output_dir: Path, run_dir: Path) -> Path:
    return _modifier_interpretability_stage_grid(
        output_dir,
        run_dir,
        stages=SUPPLEMENTAL_MODIFIER_INTERPRETABILITY_STAGES,
        stem="modifier_interpretability_early_stages.png",
        figure_size=_modifier_interpretability_figure_size(len(SUPPLEMENTAL_MODIFIER_INTERPRETABILITY_STAGES)),
        legend_ncol=2,
        legend_bbox_to_anchor=(0.5, 1.04),
    )


def _modifier_interpretability_stage_grid(
    output_dir: Path,
    run_dir: Path,
    *,
    stages: tuple[str, ...],
    stem: str,
    figure_size: tuple[float, float],
    legend_ncol: int = 1,
    legend_bbox_to_anchor: tuple[float, float] = (0.3, 0.98),
) -> Path:
    sample_frames: list[tuple[str, pd.DataFrame]] = []
    for stage_name in stages:
        sample_path = _modifier_interpretability_sample_path(run_dir, stage=stage_name)
        sample_frame = _read_csv_if_exists(sample_path)
        if sample_frame.empty:
            raise FileNotFoundError(f"Missing modifier interpretability samples: {sample_path}")
        required_columns = {"input_group", "perturbation_value", "perturbed_base_dvr_mean", "perturbed_dvr_star_mean", "stage_label"}
        missing = required_columns.difference(sample_frame.columns)
        if missing:
            raise ValueError(f"{sample_path.name} is missing required columns: {', '.join(sorted(missing))}")
        sample_frames.append((stage_name, sample_frame))

    panel_specs = [
        ("temperature", "Temperature", "Temperature offset (°C)"),
        ("daylength", "Daylength", "Daylength offset (h)"),
        ("precipitation", "Precipitation", "Precipitation multiplier (×)"),
    ]
    n_rows = len(sample_frames)
    fig, axes = plt.subplots(n_rows, 3, figsize=figure_size, sharey="row", squeeze=False)
    corrected_color = MODEL_COLORS["m1_dvr_con"]
    baseline_color = MODEL_COLORS["m0_dvr"]
    legend_handles = []
    legend_labels = []
    for row_idx, (stage_name, sample_frame) in enumerate(sample_frames):
        stage_label = str(sample_frame["stage_label"].dropna().iloc[0]) if sample_frame["stage_label"].notna().any() else stage_name
        display_stage = _modifier_stage_label_for_display(stage_label, stage_name)
        for col_idx, (input_group, title, xlabel) in enumerate(panel_specs):
            ax = axes[row_idx, col_idx]
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.2f}"))
            group = sample_frame[sample_frame["input_group"] == input_group]
            if group.empty:
                ax.set_axis_off()
                continue
            agg = (
                group.groupby("perturbation_value", as_index=False)
                .agg(
                    dvr_tp=("perturbed_base_dvr_mean", "mean"),
                    dvr_star=("perturbed_dvr_star_mean", "mean"),
                )
                .sort_values("perturbation_value")
            )
            x = agg["perturbation_value"].to_numpy(dtype=float)
            dvr_tp = agg["dvr_tp"].to_numpy(dtype=float)
            dvr_star = agg["dvr_star"].to_numpy(dtype=float)
            baseline_line = ax.plot(
                x,
                dvr_tp,
                marker="o",
                markersize=3.2,
                linewidth=1.25,
                linestyle="--",
                color=baseline_color,
                label=r"$DVR^{PT}$",
            )[0]
            corrected_line = ax.plot(
                x,
                dvr_star,
                marker="s",
                markersize=3.2,
                linewidth=1.25,
                color=corrected_color,
                label=r"$DVR^*$",
            )[0]
            if row_idx == 0 and col_idx == 0:
                legend_handles.extend([baseline_line, corrected_line])
                legend_labels.extend([r"$DVR^{PT}$", r"$DVR^*$"])
            panel_letter = chr(ord("a") + row_idx * 3 + col_idx)
            ax.text(
                0.02,
                0.96,
                f"({panel_letter})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=MANUSCRIPT_FONT_SIZES["panel"],
                fontweight="bold",
            )
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            ax.tick_params(axis="both", length=2.5, width=0.6)
            for spine_name in ("top", "right"):
                ax.spines[spine_name].set_visible(False)
        axes[row_idx, 0].set_ylabel(f"{display_stage}\nMean daily DVR")
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=False,
        ncol=legend_ncol,
        loc="upper center",
        bbox_to_anchor=legend_bbox_to_anchor,
        columnspacing=3.0,
        handlelength=3.4,
        handleheight=1.7,
        fontsize=MANUSCRIPT_FONT_SIZES["legend"] + 1.5,
    )

    plt.tight_layout()
    return _save(fig, stem, output_dir)


def _regional_model_stage_columns(
    model_ids: tuple[str, ...],
    stage_specs: tuple[tuple[str, str], ...],
) -> list[str]:
    return [
        f"{model_id}_{stage_name}_doy"
        for model_id in model_ids
        for stage_name, _ in stage_specs
    ]


def _require_regional_grid_columns(frame: pd.DataFrame, columns: list[str], *, figure_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(missing)
        raise ValueError(f"Missing columns required for {figure_name}: {missing_display}")


def _regional_scatter_map_panel(
    ax,
    china: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
    lon_values: pd.Series,
    lat_values: pd.Series,
    values: pd.Series,
    *,
    cmap: str,
    norm: Normalize,
):
    china.boundary.plot(ax=ax, color="black", linewidth=0.6)
    provinces.boundary.plot(ax=ax, color="#999999", linewidth=0.3)
    mask = values.notna() & lon_values.notna() & lat_values.notna()
    scatter = ax.scatter(
        lon_values.loc[mask],
        lat_values.loc[mask],
        c=values.loc[mask],
        cmap=cmap,
        norm=norm,
        s=8,
        marker="s",
        linewidths=0.0,
        rasterized=True,
    )
    xmin, xmax, ymin, ymax = CONFIG.figures.map_extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    # ax.grid(color="#D9D9D9", linewidth=0.25, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)
    _add_map_orientation(ax)
    return scatter


def _regional_style_map_axis(ax, *, row_index: int, col_index: int, n_rows: int) -> None:
    ax.set_xlabel("", fontsize=MANUSCRIPT_FONT_SIZES["label"])
    ax.set_ylabel("", fontsize=MANUSCRIPT_FONT_SIZES["label"])
    ax.tick_params(
        bottom=row_index == n_rows - 1,
        left=col_index == 0,
        labelbottom=row_index == n_rows - 1,
        labelleft=col_index == 0,
        labelsize=MANUSCRIPT_FONT_SIZES["tick"],
        length=2.5,
        width=0.6,
    )


def _regional_add_row_label(ax, row_label: str, *, x: float = -0.18) -> None:
    ax.annotate(
        row_label,
        xy=(x, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        rotation=90,
        fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
        fontweight="bold",
    )


def regional_heading_maturity_comparison(
    output_dir: Path,
    run_dir: Path,
    *,
    period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD,
    climatology_path: Path | str | None = None,
) -> Path:
    climatology = _load_regional_grid_climatology(run_dir, period=period, climatology_path=climatology_path)
    china, provinces = _load_boundary()
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    row_specs = (
        ("rs", "Remote sensing"),
        ("m1_dvr_con", MODEL_DISPLAY["m1_dvr_con"]),
        ("kde", ""),
    )
    required_columns = [
        f"{source_name}_{stage_name}_doy"
        for source_name in ("rs", "m1_dvr_con")
        for stage_name, _ in stage_specs
    ]
    doy_norm = Normalize(*_regional_shared_value_range(climatology, required_columns))
    lon_values = pd.to_numeric(climatology["lon"], errors="coerce")
    lat_values = pd.to_numeric(climatology["lat"], errors="coerce")

    fig, axes = plt.subplots(3, 2, figsize=MANUSCRIPT_FIGURE_SIZES["regional_comparison"], constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.005, hspace=0.005)
    axes = np.atleast_1d(axes).reshape(3, 2)
    xmin, xmax, ymin, ymax = CONFIG.figures.map_extent
    doy_scatter = None
    kde_axes = []
    panel_labels = ("a", "b", "c", "d", "e", "f")

    for row_index, (source_name, row_label) in enumerate(row_specs):
        for col_index, (stage_name, stage_label) in enumerate(stage_specs):
            ax = axes[row_index, col_index]
            if source_name == "kde":
                _regional_plot_stage_doy_kde_panel(
                    ax,
                    climatology,
                    stage_name,
                    show_legend=(col_index == len(stage_specs) - 1),
                )
                ax.set_xlabel("Day of year (DOY)", fontsize=MANUSCRIPT_FONT_SIZES["label"])
                ax.set_ylabel("Density (x10^-2)" if col_index == 0 else "", fontsize=MANUSCRIPT_FONT_SIZES["label"])
                ax.set_title("", fontsize=MANUSCRIPT_FONT_SIZES["panel"], fontweight="bold")
                ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"], length=2.5, width=0.6)
                # ax.grid(color="#D9D9D9", linewidth=0.3, alpha=0.65)
                kde_axes.append(ax)
            else:
                china.boundary.plot(ax=ax, color="black", linewidth=0.6)
                provinces.boundary.plot(ax=ax, color="#999999", linewidth=0.3)
                values = pd.to_numeric(climatology[f"{source_name}_{stage_name}_doy"], errors="coerce")
                mask = values.notna() & lon_values.notna() & lat_values.notna()
                doy_scatter = ax.scatter(
                    lon_values.loc[mask],
                    lat_values.loc[mask],
                    c=values.loc[mask],
                    cmap="cividis",
                    norm=doy_norm,
                    s=8,
                    marker="s",
                    linewidths=0.0,
                    rasterized=True,
                )
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("", fontsize=MANUSCRIPT_FONT_SIZES["label"])
                ax.set_ylabel("", fontsize=MANUSCRIPT_FONT_SIZES["label"])
                ax.set_title(stage_label if row_index == 0 else "", fontsize=MANUSCRIPT_FONT_SIZES["panel"], fontweight="bold")
                ax.tick_params(
                    bottom=row_index == 1,
                    left=col_index == 0,
                    labelbottom=row_index == 1,
                    labelleft=col_index == 0,
                    labelsize=MANUSCRIPT_FONT_SIZES["tick"],
                    length=2.5,
                    width=0.6,
                )
                # ax.grid(color="#D9D9D9", linewidth=0.25, alpha=0.55)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                _add_map_orientation(ax)

            _regional_add_panel_label(ax, panel_labels[row_index * len(stage_specs) + col_index])
            if col_index == 0 and row_label:
                ax.annotate(
                    row_label,
                    xy=(-0.18, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
                    fontweight="bold",
                )

    if kde_axes:
        density_top = max(ax.get_ylim()[1] for ax in kde_axes)
        density_top = max(0.01, float(np.ceil(density_top / 0.01) * 0.01))
        for ax in kde_axes:
            ax.set_ylim(0.0, density_top)
        for ax in kde_axes[1:]:
            ax.tick_params(axis="y", left=False, labelleft=False)

    if doy_scatter is not None:
        colorbar = fig.colorbar(
            doy_scatter,
            ax=axes[:2, :].ravel().tolist(),
            orientation="vertical",
            location="right",
            shrink=1.0,
            aspect=30,
            pad=0.015,
            label="Day of year (DOY)",
        )
        colorbar.ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"])
        colorbar.set_label("Day of year (DOY)", fontsize=MANUSCRIPT_FONT_SIZES["label"])
    return _save(fig, "regional_heading_maturity_comparison.png", output_dir)


def regional_residual_diagnostics(
    output_dir: Path,
    run_dir: Path,
    *,
    period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD,
    climatology_path: Path | str | None = None,
) -> Path:
    climatology = _load_regional_grid_climatology(run_dir, period=period, climatology_path=climatology_path)
    china, provinces = _load_boundary()
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    residuals_by_stage = {stage_name: _regional_stage_residual_values(climatology, stage_name) for stage_name, _ in stage_specs}
    residual_limit = _regional_residual_limit(climatology, stage_specs)
    residual_norm = TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit)
    x_limit = _regional_residual_distribution_limit(tuple(residuals_by_stage.values()))
    x_grid = np.linspace(-x_limit, x_limit, 500)
    lon_values = pd.to_numeric(climatology["lon"], errors="coerce")
    lat_values = pd.to_numeric(climatology["lat"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=MANUSCRIPT_FIGURE_SIZES["regional_residual"], constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(2, 2)
    xmin, xmax, ymin, ymax = CONFIG.figures.map_extent
    residual_scatter = None
    residual_density_axes = []
    panel_labels = ("a", "b", "c", "d")

    for col_index, (stage_name, stage_label) in enumerate(stage_specs):
        ax = axes[0, col_index]
        china.boundary.plot(ax=ax, color="black", linewidth=0.6)
        provinces.boundary.plot(ax=ax, color="#999999", linewidth=0.3)
        values = (
            pd.to_numeric(climatology[f"m1_dvr_con_{stage_name}_doy"], errors="coerce")
            - pd.to_numeric(climatology[f"rs_{stage_name}_doy"], errors="coerce")
        )
        mask = values.notna() & lon_values.notna() & lat_values.notna()
        residual_scatter = ax.scatter(
            lon_values.loc[mask],
            lat_values.loc[mask],
            c=values.loc[mask],
            cmap="RdBu_r",
            norm=residual_norm,
            s=8,
            marker="s",
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(stage_label, fontsize=MANUSCRIPT_FONT_SIZES["panel"], fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=MANUSCRIPT_FONT_SIZES["label"])
        ax.set_ylabel("Latitude" if col_index == 0 else "", fontsize=MANUSCRIPT_FONT_SIZES["label"])
        ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"], length=2.5, width=0.6)
        # ax.grid(color="#D9D9D9", linewidth=0.25, alpha=0.55)
        _add_map_orientation(ax)
        _regional_add_panel_label(ax, panel_labels[col_index])

        ax = axes[1, col_index]
        _regional_plot_residual_distribution_panel(
            ax,
            residuals_by_stage[stage_name],
            x_grid=x_grid,
            x_limit=x_limit,
        )
        ax.set_title("", fontsize=MANUSCRIPT_FONT_SIZES["panel"], fontweight="bold")
        ax.set_xlabel("Prediction - remote sensing (days)", fontsize=MANUSCRIPT_FONT_SIZES["label"])
        ax.set_ylabel("Density (x10^-2)" if col_index == 0 else "", fontsize=MANUSCRIPT_FONT_SIZES["label"])
        _regional_add_panel_label(ax, panel_labels[col_index + 2])
        residual_density_axes.append(ax)

    if residual_density_axes:
        density_top = max(ax.get_ylim()[1] for ax in residual_density_axes)
        density_top = max(0.01, float(np.ceil(density_top / 0.01) * 0.01))
        for ax in residual_density_axes:
            ax.set_ylim(0.0, density_top)
        for ax in residual_density_axes[1:]:
            ax.tick_params(axis="y", left=False, labelleft=False)

    if residual_scatter is not None:
        colorbar = fig.colorbar(
            residual_scatter,
            ax=axes[0, :].tolist(),
            orientation="vertical",
            location="right",
            shrink=1.0,
            aspect=30,
            pad=0.025,
            extend="both",
            label=f"Residual ({MODEL_DISPLAY['m1_dvr_con']} - remote sensing, days)",
        )
        colorbar.ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"])
        colorbar.set_label(f"Residual(days)", fontsize=MANUSCRIPT_FONT_SIZES["label"])

    return _save(fig, "regional_residual_diagnostics.png", output_dir)


def regional_supplemental_model_heading_maturity_comparison(
    output_dir: Path,
    run_dir: Path,
    *,
    period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD,
    climatology_path: Path | str | None = None,
) -> Path:
    climatology = _load_regional_grid_climatology(run_dir, period=period, climatology_path=climatology_path)
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    source_specs = (
        ("rs", "Remote sensing"),
        *[(model_id, MODEL_DISPLAY[model_id]) for model_id in SUPPLEMENTAL_REGIONAL_MODEL_IDS],
    )
    required_columns = [
        f"rs_{stage_name}_doy" for stage_name, _ in stage_specs
    ] + _regional_model_stage_columns(SUPPLEMENTAL_REGIONAL_MODEL_IDS, stage_specs)
    _require_regional_grid_columns(
        climatology,
        required_columns,
        figure_name="regional supplemental model heading/maturity comparison",
    )
    china, provinces = _load_boundary()
    doy_norm = Normalize(*_regional_shared_value_range(climatology, required_columns))
    lon_values = pd.to_numeric(climatology["lon"], errors="coerce")
    lat_values = pd.to_numeric(climatology["lat"], errors="coerce")

    fig, axes = plt.subplots(
        4,
        2,
        figsize=MANUSCRIPT_FIGURE_SIZES["regional_supplemental_comparison"],
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.005, hspace=0.005)
    axes = np.atleast_1d(axes).reshape(4, 2)
    panel_labels = tuple(chr(ord("a") + index) for index in range(axes.size))
    doy_scatter = None

    for row_index, (source_name, row_label) in enumerate(source_specs):
        for col_index, (stage_name, stage_label) in enumerate(stage_specs):
            ax = axes[row_index, col_index]
            values = pd.to_numeric(climatology[f"{source_name}_{stage_name}_doy"], errors="coerce")
            doy_scatter = _regional_scatter_map_panel(
                ax,
                china,
                provinces,
                lon_values,
                lat_values,
                values,
                cmap="cividis",
                norm=doy_norm,
            )
            ax.set_title(
                stage_label if row_index == 0 else "",
                fontsize=MANUSCRIPT_FONT_SIZES["panel"],
                fontweight="bold",
            )
            _regional_style_map_axis(ax, row_index=row_index, col_index=col_index, n_rows=len(source_specs))
            _regional_add_panel_label(ax, panel_labels[row_index * len(stage_specs) + col_index])
            if col_index == 0:
                _regional_add_row_label(ax, row_label)

    if doy_scatter is not None:
        colorbar = fig.colorbar(
            doy_scatter,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            location="bottom",
            shrink=0.88,
            aspect=36,
            pad=0.025,
            label="Day of year (DOY)",
        )
        colorbar.ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"])
        colorbar.set_label("Day of year (DOY)", fontsize=MANUSCRIPT_FONT_SIZES["label"])

    return _save(fig, "regional_supplemental_model_heading_maturity_comparison.png", output_dir)


def regional_supplemental_model_residual_diagnostics(
    output_dir: Path,
    run_dir: Path,
    *,
    period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD,
    climatology_path: Path | str | None = None,
) -> Path:
    climatology = _load_regional_grid_climatology(run_dir, period=period, climatology_path=climatology_path)
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    model_specs = tuple((model_id, MODEL_DISPLAY[model_id]) for model_id in SUPPLEMENTAL_REGIONAL_MODEL_IDS)
    required_columns = [
        f"rs_{stage_name}_doy" for stage_name, _ in stage_specs
    ] + _regional_model_stage_columns(SUPPLEMENTAL_REGIONAL_MODEL_IDS, stage_specs)
    _require_regional_grid_columns(
        climatology,
        required_columns,
        figure_name="regional supplemental model residual diagnostics",
    )
    china, provinces = _load_boundary()
    residual_limit = _regional_residual_limit(
        climatology,
        stage_specs,
        model_ids=SUPPLEMENTAL_REGIONAL_MODEL_IDS,
    )
    residual_norm = TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit)
    lon_values = pd.to_numeric(climatology["lon"], errors="coerce")
    lat_values = pd.to_numeric(climatology["lat"], errors="coerce")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=MANUSCRIPT_FIGURE_SIZES["regional_supplemental_residual"],
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.005, hspace=0.005)
    axes = np.atleast_1d(axes).reshape(3, 2)
    panel_labels = tuple(chr(ord("a") + index) for index in range(axes.size))
    residual_scatter = None

    for row_index, (model_id, row_label) in enumerate(model_specs):
        for col_index, (stage_name, stage_label) in enumerate(stage_specs):
            ax = axes[row_index, col_index]
            values = (
                pd.to_numeric(climatology[f"{model_id}_{stage_name}_doy"], errors="coerce")
                - pd.to_numeric(climatology[f"rs_{stage_name}_doy"], errors="coerce")
            )
            residual_scatter = _regional_scatter_map_panel(
                ax,
                china,
                provinces,
                lon_values,
                lat_values,
                values,
                cmap="RdBu_r",
                norm=residual_norm,
            )
            ax.set_title(
                stage_label if row_index == 0 else "",
                fontsize=MANUSCRIPT_FONT_SIZES["panel"],
                fontweight="bold",
            )
            _regional_style_map_axis(ax, row_index=row_index, col_index=col_index, n_rows=len(model_specs))
            _regional_add_panel_label(ax, panel_labels[row_index * len(stage_specs) + col_index])
            if col_index == 0:
                _regional_add_row_label(ax, row_label)

    if residual_scatter is not None:
        colorbar = fig.colorbar(
            residual_scatter,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            location="bottom",
            shrink=0.88,
            aspect=30,
            pad=0.02,
            extend="both",
            label="Prediction - remote sensing (days)",
        )
        colorbar.ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"])
        colorbar.set_label("Prediction - remote sensing (days)", fontsize=MANUSCRIPT_FONT_SIZES["label"])

    return _save(fig, "regional_supplemental_model_residual_diagnostics.png", output_dir)


def regional_reviving_offset_sensitivity(
    output_dir: Path,
    *,
    metrics_path: Path | str | None = None,
    sensitivity_run_id: str = DEFAULT_REGIONAL_REVIVING_OFFSET_SENSITIVITY_RUN_ID,
    reference_offset_days: float = REGIONAL_REVIVING_OFFSET_REFERENCE_DAYS,
) -> Path:
    resolved_metrics_path = (
        Path(metrics_path)
        if metrics_path is not None
        else _regional_reviving_offset_sensitivity_metrics_path(sensitivity_run_id=sensitivity_run_id)
    )
    metrics = _load_regional_reviving_offset_sensitivity_metrics(resolved_metrics_path)
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    metric_specs = (("mae", "MAE", "MAE (days)"), ("bias", "Bias", "Bias (days)"))
    model_ids = tuple(model_id for model_id in _comparison_models() if model_id in set(metrics["model"]))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=MANUSCRIPT_FIGURE_SIZES["regional_reviving_offset_sensitivity"],
        sharex=True,
        squeeze=False,
    )
    axes = np.atleast_1d(axes).reshape(2, 2)
    panel_labels = tuple(chr(ord("a") + index) for index in range(axes.size))
    legend_handles = []
    legend_labels = []

    for row_index, (stage_name, stage_label) in enumerate(stage_specs):
        for col_index, (metric_name, metric_label, metric_axis_label) in enumerate(metric_specs):
            ax = axes[row_index, col_index]
            for model_id in model_ids:
                subset = metrics[(metrics["stage"] == stage_name) & (metrics["model"] == model_id)].sort_values(
                    "reviving_offset_days"
                )
                if subset.empty:
                    continue
                handle = ax.plot(
                    subset["reviving_offset_days"],
                    subset[metric_name],
                    color=MODEL_COLORS[model_id],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.45,
                    label=MODEL_DISPLAY[model_id],
                )[0]
                if row_index == 0 and col_index == 0:
                    legend_handles.append(handle)
                    legend_labels.append(MODEL_DISPLAY[model_id])

            ax.axvline(reference_offset_days, color="#4D4D4D", linestyle=":", linewidth=1.0)
            if metric_name == "bias":
                ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
            ax.text(
                0.02,
                0.96,
                f"({panel_labels[row_index * len(metric_specs) + col_index]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
            )
            if row_index == 0:
                ax.set_title(metric_label, fontsize=MANUSCRIPT_FONT_SIZES["panel"], fontweight="bold")
            if row_index == len(stage_specs) - 1:
                ax.set_xlabel("Reviving offset after transplanting (days)", fontsize=MANUSCRIPT_FONT_SIZES["label"])
            ylabel = f"{stage_label}\n{metric_axis_label}" if col_index == 0 else metric_axis_label
            ax.set_ylabel(ylabel, fontsize=MANUSCRIPT_FONT_SIZES["label"])
            ax.grid(color="#D9D9D9", linewidth=0.35, alpha=0.65)
            ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"], length=2.5, width=0.6)
            for spine_name in ("top", "right"):
                ax.spines[spine_name].set_visible(False)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            frameon=False,
            ncol=len(legend_labels),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=MANUSCRIPT_FONT_SIZES["legend"],
            columnspacing=1.4,
            handlelength=2.0,
        )

    plt.tight_layout()
    return _save(fig, REGIONAL_REVIVING_OFFSET_SENSITIVITY_FIGURE_FILENAME, output_dir)


def _load_regional_reviving_offset_sensitivity_metrics(metrics_path: Path | str) -> pd.DataFrame:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing regional reviving offset sensitivity metrics: {path}")

    metrics = pd.read_csv(path)
    required_columns = {"reviving_offset_days", "model", "stage", "mae", "bias"}
    missing = required_columns.difference(metrics.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing required columns: {missing_display}")

    metrics = metrics[metrics["stage"].isin(("heading", "maturity"))].copy()
    metrics = metrics[metrics["model"].isin(_comparison_models())].copy()
    for column in ("reviving_offset_days", "mae", "bias"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics.dropna(subset=["reviving_offset_days", "mae", "bias"])
    if metrics.empty:
        raise ValueError(f"{path.name} does not contain finite heading/maturity sensitivity metrics")
    return metrics.sort_values(["stage", "model", "reviving_offset_days"]).reset_index(drop=True)


def regional_paired_scatter_diagnostics(
    output_dir: Path,
    run_dir: Path,
    *,
    period: str = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD,
    climatology_path: Path | str | None = None,
) -> Path:
    climatology = _load_regional_grid_climatology(run_dir, period=period, climatology_path=climatology_path)
    stage_specs = (("heading", "Heading"), ("maturity", "Maturity"))
    model_label = MODEL_DISPLAY["m1_dvr_con"]

    fig, axes = plt.subplots(1, 2, figsize=MANUSCRIPT_FIGURE_SIZES["regional_paired"], constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(1, 2)
    for col_index, (stage_name, stage_label) in enumerate(stage_specs):
        ax = axes[0, col_index]
        _regional_plot_stage_scatter_panel(ax, climatology, stage_name, stage_label, model_label)
        _regional_add_panel_label(ax, ("a", "b")[col_index])

    return _save(fig, "regional_paired_scatter_diagnostics.png", output_dir)


def _regional_period_display(period: str) -> str:
    return period.replace("_", "-")


def _regional_add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
    )


def _regional_plot_stage_doy_kde_panel(ax, frame: pd.DataFrame, stage_name: str, *, show_legend: bool = True) -> None:
    obs = pd.to_numeric(frame[f"rs_{stage_name}_doy"], errors="coerce").dropna().to_numpy(dtype=float)
    pred = pd.to_numeric(frame[f"m1_dvr_con_{stage_name}_doy"], errors="coerce").dropna().to_numpy(dtype=float)
    x_grid, x_limits = _regional_distribution_grid((obs, pred), step=5.0, pad=5.0)
    series_specs = (
        ("Remote sensing", obs, "#5F6368", 0.18),
        (MODEL_DISPLAY["m1_dvr_con"], pred, MODEL_COLORS["m1_dvr_con"], 0.22),
    )
    for label, values, color, alpha in series_specs:
        density = _regional_kernel_density(values, x_grid)
        if density is None:
            continue
        ax.fill_between(x_grid, 0.0, density, color=color, alpha=alpha, linewidth=0.0)
        ax.plot(x_grid, density, color=color, linewidth=1.65, label=label)

    obs_mean = _regional_finite_mean(obs)
    pred_mean = _regional_finite_mean(pred)
    if np.isfinite(obs_mean):
        ax.axvline(obs_mean, color="#5F6368", linestyle="--", linewidth=1.15)
    if np.isfinite(pred_mean):
        ax.axvline(pred_mean, color=MODEL_COLORS["m1_dvr_con"], linestyle="--", linewidth=1.15)
    ax.text(
        0.97,
        0.96,
        f"RS mean = {_regional_format_days(obs_mean)}\nModel mean = {_regional_format_days(pred_mean)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.88, "pad": 2.4},
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value * 100.0:g}"))
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.4, 0.86), fontsize=MANUSCRIPT_FONT_SIZES["legend"], frameon=True, framealpha=0.82)


def _regional_plot_residual_distribution_panel(
    ax,
    values: np.ndarray,
    *,
    x_grid: np.ndarray,
    x_limit: float,
) -> None:
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    if len(values) == 0:
        ax.text(0.5, 0.5, "No finite residuals", transform=ax.transAxes, ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["annotation"])
    else:
        bins = _regional_residual_histogram_bins(values)
        p5, median, p95 = np.quantile(values, [0.05, 0.50, 0.95])
        mean = float(np.mean(values))
        ax.axvspan(p5, p95, color="#8FB3D9", alpha=0.18)
        ax.hist(values, bins=bins, density=True, color="#9E9E9E", edgecolor="white", linewidth=0.35, alpha=0.55)
        kde_values = _regional_kernel_density(values, x_grid)
        if kde_values is not None:
            ax.plot(x_grid, kde_values, color="#1F4E79", linewidth=1.8)
        ax.axvline(p5, color="#777777", linestyle=":", linewidth=1.0)
        ax.axvline(p95, color="#777777", linestyle=":", linewidth=1.0)
        ax.axvline(median, color="#1F4E79", linestyle="--", linewidth=1.35)
        ax.axvline(mean, color="#B65A2A", linestyle="-.", linewidth=1.25)
        ax.text(
            0.97,
            0.96,
            f"mean = {mean:.1f} d\nmedian = {median:.1f} d",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.88, "pad": 2.4},
        )
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value * 100.0:g}"))
    # ax.grid(color="#D9D9D9", linewidth=0.35, alpha=0.65)
    ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"])


def _regional_plot_stage_scatter_panel(
    ax,
    frame: pd.DataFrame,
    stage_name: str,
    stage_label: str,
    model_label: str,
    *,
    model_id: str = "m1_dvr_con",
) -> None:
    obs, pred = _regional_stage_pair_values(frame, stage_name, model_id=model_id)
    lower, upper = _regional_paired_axis_limits(obs, pred)
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.0, label="1:1")

    if len(obs):
        ax.scatter(obs, pred, s=9, color=MODEL_COLORS[model_id], alpha=0.32, edgecolors="none", rasterized=True)
        if len(obs) >= 2 and not np.isclose(float(np.std(obs)), 0.0):
            slope, intercept = np.polyfit(obs, pred, 1)
            x_values = np.array([lower, upper], dtype=float)
            ax.plot(x_values, slope * x_values + intercept, color="red", linewidth=1.25, label="Linear fit")
    else:
        ax.text(0.5, 0.5, "No finite paired values", transform=ax.transAxes, ha="center", va="center", fontsize=MANUSCRIPT_FONT_SIZES["annotation"])

    metrics = _regional_stage_metric_summary(frame, stage_name, model_id=model_id)
    ax.text(
        0.04,
        0.88,
        (
            f"MAE = {_regional_format_days(float(metrics['mae']))}\n"
            f"RMSE = {_regional_format_days(float(metrics['rmse']))}\n"
            f"Bias = {_regional_format_signed_days(float(metrics['bias']))}\n"
            f"R2 = {_regional_format_unitless(float(metrics['r2']))}\n"
            f"n = {int(metrics['n'])}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=MANUSCRIPT_FONT_SIZES["annotation"],
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.88, "pad": 2.4},
    )
    ax.set_title(stage_label, fontsize=MANUSCRIPT_FONT_SIZES["panel"], fontweight="bold")
    ax.set_xlabel("Remote sensing DOY", fontsize=MANUSCRIPT_FONT_SIZES["label"])
    ax.set_ylabel(f"{model_label} DOY", fontsize=MANUSCRIPT_FONT_SIZES["label"])
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    # ax.grid(color="#D9D9D9", linewidth=0.35, alpha=0.65)
    ax.tick_params(labelsize=MANUSCRIPT_FONT_SIZES["tick"], length=2.5, width=0.6)
    ax.legend(loc="lower right", fontsize=MANUSCRIPT_FONT_SIZES["legend"], frameon=True, framealpha=0.82)


def _regional_stage_metric_summary(frame: pd.DataFrame, stage_name: str, *, model_id: str = "m1_dvr_con") -> dict[str, float | int]:
    obs, pred = _regional_stage_pair_values(frame, stage_name, model_id=model_id)
    if not len(obs):
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan"), "n": 0}
    diff = pred - obs
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r2": _regional_r2(obs, pred),
        "n": int(len(obs)),
    }


def _regional_stage_pair_values(frame: pd.DataFrame, stage_name: str, *, model_id: str = "m1_dvr_con") -> tuple[np.ndarray, np.ndarray]:
    obs = pd.to_numeric(frame[f"rs_{stage_name}_doy"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(frame[f"{model_id}_{stage_name}_doy"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def _regional_stage_residual_values(frame: pd.DataFrame, stage_name: str, *, model_id: str = "m1_dvr_con") -> np.ndarray:
    obs, pred = _regional_stage_pair_values(frame, stage_name, model_id=model_id)
    return pred - obs


def _regional_distribution_grid(
    value_groups: tuple[np.ndarray, ...],
    *,
    step: float,
    pad: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    values = [values[np.isfinite(values)] for values in value_groups if len(values)]
    if not values:
        return np.linspace(0.0, 1.0, 100), (0.0, 1.0)
    combined = np.concatenate(values)
    lower = float(np.floor((combined.min() - pad) / step) * step)
    upper = float(np.ceil((combined.max() + pad) / step) * step)
    if not np.isfinite(lower) or not np.isfinite(upper) or np.isclose(lower, upper):
        upper = lower + step
    return np.linspace(lower, upper, 500), (lower, upper)


def _regional_paired_axis_limits(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    _, limits = _regional_distribution_grid((obs, pred), step=5.0, pad=5.0)
    return limits


def _regional_kernel_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or np.isclose(float(np.nanstd(finite)), 0.0):
        return None
    return _evaluate_kernel_density(finite, x_grid)


def _evaluate_kernel_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if _scipy_gaussian_kde is not None:
        return _scipy_gaussian_kde(values)(x_grid)
    std = float(np.nanstd(values, ddof=1))
    bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max(std, 1.0)
    scaled = (x_grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * scaled**2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))


def _regional_shared_value_range(frame: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    series = [
        pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        for column in columns
        if column in frame.columns
    ]
    if not series:
        return 0.0, 1.0
    values = np.concatenate(series)
    if len(values) == 0:
        return 0.0, 1.0
    lower = float(np.floor(values.min()))
    upper = float(np.ceil(values.max()))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def _regional_residual_limit(
    frame: pd.DataFrame,
    stage_specs: tuple[tuple[str, str], ...],
    *,
    model_ids: tuple[str, ...] = ("m1_dvr_con",),
) -> float:
    residuals = []
    for model_id in model_ids:
        for stage_name, _ in stage_specs:
            residual = _regional_stage_residual_values(frame, stage_name, model_id=model_id)
            if len(residual):
                residuals.append(residual)
    if not residuals:
        return 1.0
    limit = float(np.ceil(np.quantile(np.abs(np.concatenate(residuals)), 0.99)))
    if not np.isfinite(limit) or np.isclose(limit, 0.0):
        return 1.0
    return limit


def _regional_residual_distribution_limit(residual_groups: tuple[np.ndarray, ...]) -> float:
    values = [np.abs(values) for values in residual_groups if len(values)]
    if not values:
        return 30.0
    quantile = float(np.quantile(np.concatenate(values), 0.995))
    limit = float(np.ceil(quantile / 5.0) * 5.0)
    if not np.isfinite(limit) or limit <= 0.0:
        return 30.0
    return max(30.0, limit)


def _regional_residual_histogram_bins(values: np.ndarray) -> np.ndarray:
    lower = float(np.floor(np.nanmin(values) / 5.0) * 5.0)
    upper = float(np.ceil(np.nanmax(values) / 5.0) * 5.0)
    if not np.isfinite(lower) or not np.isfinite(upper) or np.isclose(lower, upper):
        lower, upper = -5.0, 5.0
    return np.arange(lower, upper + 5.0, 5.0)


def _regional_finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _regional_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2:
        return float("nan")
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    ss_res = float(np.sum((pred - obs) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _regional_format_days(value: float) -> str:
    return f"{value:.1f} d" if np.isfinite(value) else "NA"


def _regional_format_signed_days(value: float) -> str:
    return f"{value:+.1f} d" if np.isfinite(value) else "NA"


def _regional_format_unitless(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "NA"


def sample_distributions(output_dir: Path, eval_dir: Path | None = None) -> Path:
    modeling_df = build_modeling_dataset(force=False)
    by_year = modeling_df.groupby("year", as_index=False).size().rename(columns={"size": "n"}).sort_values("year")
    by_site = modeling_df.groupby("SID", as_index=False).size().rename(columns={"size": "n"}).sort_values("SID")

    fig, axes = plt.subplots(1, 2, figsize=MANUSCRIPT_FIGURE_SIZES["sample_distribution"])
    axes[0].bar(by_year["year"].astype(str), by_year["n"], color=FIGURE_STYLE.primary, alpha=0.85)
    axes[0].set_title("Samples by year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Sample count")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(by_site["SID"].astype(str), by_site["n"], color=FIGURE_STYLE.secondary, alpha=0.85)
    axes[1].set_title("Samples by site")
    axes[1].set_xlabel("Station ID")
    axes[1].set_ylabel("Sample count")
    return _save(fig, "sample_distributions.png", output_dir)

def temporal_coverage_by_site_year(output_dir: Path) -> Path:
    _, _, _, coverage = _load_study_area_inputs()

    observed = (coverage.to_numpy(dtype=float) > 0.0).astype(float)
    year_counts = observed.sum(axis=0)
    site_counts = observed.sum(axis=1)
    n_sites, n_years = observed.shape
    observed_color = "#176D75"
    missing_color = "#F4EFE1"
    marginal_color = "#B87333"
    grid_color = "#FFFFFF"

    fig = plt.figure(figsize=MANUSCRIPT_FIGURE_SIZES["coverage"], constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(18, 2.7),
        height_ratios=(2.2, 18),
        wspace=0.08,
        hspace=0.06,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)

    ax_top.bar(np.arange(n_years), year_counts, color=marginal_color, width=0.82, linewidth=0.0)
    ax_top.set_xlim(-0.5, n_years - 0.5)
    ax_top.set_ylabel("Sites")
    ax_top.set_xticks([])
    ax_top.grid(axis="y", color="#D8D2C4", linewidth=0.5, alpha=0.8)
    ax_top.spines[["top", "right"]].set_visible(False)
    ax_top.tick_params(axis="y", length=2.5, width=0.6)

    coverage_cmap = ListedColormap([missing_color, observed_color])
    ax.imshow(observed, aspect="auto", interpolation="none", cmap=coverage_cmap, vmin=0.0, vmax=1.0)
    year_step = max(1, int(np.ceil(n_years / 14)))
    year_tick_positions = np.arange(0, n_years, year_step)
    site_step = max(1, int(np.ceil(n_sites / 24)))
    site_tick_positions = np.arange(0, n_sites, site_step)
    ax.set_xticks(year_tick_positions)
    ax.set_xticklabels(coverage.columns[year_tick_positions], rotation=45, ha="right")
    ax.set_yticks(site_tick_positions)
    ax.set_yticklabels(coverage.index[site_tick_positions])
    ax.set_xticks(np.arange(-0.5, n_years, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_sites, 1.0), minor=True)
    ax.grid(which="minor", color=grid_color, linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Year")
    ax.set_ylabel("Station ID")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax_right.barh(np.arange(n_sites), site_counts, color=marginal_color, height=0.72, linewidth=0.0)
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_xlabel("Years")
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.grid(axis="x", color="#D8D2C4", linewidth=0.5, alpha=0.8)
    ax_right.spines[["top", "right"]].set_visible(False)

    ax.legend(
        handles=[Patch(facecolor=observed_color, edgecolor="none", label="Observed site-year")],
        loc="lower right",
        # bbox_to_anchor=(0.2, 0.05),
        bbox_transform=ax.transData,
        frameon=True,
        framealpha=0.92,
        facecolor=missing_color,
        edgecolor="none",
        fontsize=MANUSCRIPT_FONT_SIZES["legend"],
        handlelength=1.6,
        handletextpad=0.5,
        borderpad=0.35,
    )
    return _save(fig, "temporal_coverage_by_site_year.png", output_dir)

def threshold_heterogeneity(output_dir: Path) -> Path:
    panel_data = threshold_heterogeneity_panel_data()
    distribution_long = panel_data["distribution_long"]
    counts = panel_data["counts"]
    stage_labels = threshold_heterogeneity_stage_labels()

    fig, axes = plt.subplots(2, 3, figsize=MANUSCRIPT_FIGURE_SIZES["threshold_heterogeneity"], sharex=False)
    column_specs = [
        ("site_year", FIGURE_STYLE.primary, f"A. Site-year heterogeneity\n(valid site-year samples, n={counts['site_year']})"),
        ("spatial", FIGURE_STYLE.secondary, f"B. Spatial heterogeneity\n(site samples, n={counts['spatial']})"),
        ("temporal", FIGURE_STYLE.accent, f"C. Interannual heterogeneity\n(year samples, n={counts['temporal']})"),
    ]
    row_specs = [
        ("days_after_transplanting", "box", "Days after transplanting"),
        ("cumulative_thermal_requirement", "violin", "Cumulative thermal requirement from transplanting"),
    ]
    for col_idx, (scope, color, title) in enumerate(column_specs):
        axes[0, col_idx].set_title(title)
        for row_idx, (metric, kind, ylabel) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            subset = distribution_long[
                (distribution_long["scope"] == scope) & (distribution_long["metric"] == metric)
            ].copy()
            _draw_distribution_panel(ax, _stage_value_arrays(subset, FIGURE4_STAGE_ORDER), kind=kind, color=color)
            ax.set_xticks(range(1, len(FIGURE4_STAGE_ORDER) + 1))
            ax.set_xticklabels(stage_labels)
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 1:
                ax.set_xlabel("Stage")
            else:
                ax.set_xlabel("")

    return _save(fig, "threshold_heterogeneity.png", output_dir)

def obs_vs_simulated(output_dir: Path, run_dir: Path) -> Path:
    representative = _load_global_representative_seed_prediction_long(run_dir, stage="maturity")
    model_order = _comparison_models()
    focus_stage = "maturity"
    plot_data = representative[
        representative["task"].isin(MAIN_DVR_TASKS)
        & representative["model"].isin(model_order)
        & (representative["stage"] == focus_stage)
    ].copy()
    axis_values = pd.concat(
        [
            pd.to_numeric(plot_data["obs_doy"], errors="coerce"),
            pd.to_numeric(plot_data["pred_doy"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if axis_values.empty:
        lower, upper = 150.0, 350.0
    else:
        lower = float(np.floor((axis_values.min() - 6.0) / 25.0) * 25.0)
        upper = float(np.ceil((axis_values.max() + 6.0) / 25.0) * 25.0)
        if upper <= lower:
            upper = lower + 25.0
    tick_step = 50.0 if upper - lower > 125.0 else 25.0
    axis_ticks = np.arange(lower, upper + 0.5 * tick_step, tick_step)

    fig, axes = plt.subplots(
        len(model_order),
        len(MAIN_DVR_TASKS),
        figsize=MANUSCRIPT_FIGURE_SIZES["obs_sim_matrix"],
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    legend_handles = []
    legend_labels = []
    for row_idx, model in enumerate(model_order):
        for col_idx, task in enumerate(MAIN_DVR_TASKS):
            ax = axes[row_idx, col_idx]
            subset = plot_data[(plot_data["task"] == task) & (plot_data["model"] == model)].dropna(
                subset=["obs_doy", "pred_doy"]
            )
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.set_aspect("equal", adjustable="box")
            # ax.set_facecolor("#FAFAF7")
            if subset.empty:
                ax.set_axis_off()
                continue
            fold_values = (
                sorted(subset["fold"].dropna().unique())
                if "fold" in subset.columns and subset["fold"].notna().any()
                else [np.nan]
            )
            for fold in fold_values:
                fold_subset = subset[subset["fold"] == fold] if not pd.isna(fold) and "fold" in subset.columns else subset
                if fold_subset.empty:
                    continue
                scatter = ax.scatter(
                    fold_subset["obs_doy"],
                    fold_subset["pred_doy"],
                    s=13,
                    alpha=0.72,
                    color=_fold_color(fold),
                    edgecolors="white",
                    linewidths=0.25,
                    rasterized=True,
                    label=f"Fold {int(fold)}" if not pd.isna(fold) else STAGE_DISPLAY[focus_stage],
                )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(scatter)
                    legend_labels.append(f"Fold {int(fold)}" if not pd.isna(fold) else STAGE_DISPLAY[focus_stage])
            ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.1, color="#1F1F1F")
            mae = _mae(subset["obs_doy"], subset["pred_doy"])
            rmse = _rmse(subset["obs_doy"], subset["pred_doy"])
            r2 = _r2(subset["obs_doy"], subset["pred_doy"])
            ax.text(
                0.03,
                0.97,
                f"MAE {mae:.2f}\nRMSE {rmse:.2f}\nR² {r2:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.0,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
            )
            if row_idx == 0:
                ax.set_title(TASK_DISPLAY[task], fontsize=MANUSCRIPT_FONT_SIZES["panel"], fontweight="bold")
            if col_idx == 0:
                ax.text(
                    -0.32,
                    0.5,
                    MODEL_DISPLAY[model],
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=MANUSCRIPT_FONT_SIZES["label"],
                    fontweight="bold",
                )
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(
                labelbottom=row_idx == len(model_order) - 1,
                labelleft=col_idx == 0,
                labelsize=MANUSCRIPT_FONT_SIZES["tick"],
                length=2.5,
                width=0.6,
            )
            # ax.grid(color="#D9D9D9", linewidth=0.35, alpha=0.55)
            for spine in ax.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#333333")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            fontsize=7.5,
            columnspacing=1.2,
            handletextpad=0.4,
        )
    fig.supxlabel("Observed DOY", fontsize=MANUSCRIPT_FONT_SIZES["label"], y=0.035)
    fig.supylabel("Simulated DOY", fontsize=MANUSCRIPT_FONT_SIZES["label"], x=0.012)
    if hasattr(fig, "subplots_adjust"):
        fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.08, wspace=0.14, hspace=0.14)
    return _save(fig, "obs_vs_simulated.png", output_dir)

def regional_weather_climatology(output_dir: Path) -> Path:
    period = DEFAULT_REGIONAL_GRID_FIGURE_PERIOD
    climatology = _load_regional_climatology_table(period=period)
    china, provinces = _load_boundary()
    panels = [
        ("TemMin_climatology", "Mean annual TemMin (°C)", "coolwarm"),
        ("TemMax_climatology", "Mean annual TemMax (°C)", "coolwarm"),
        ("TemAver_climatology", "Mean annual TemAver (°C)", "coolwarm"),
        ("Precipitation_climatology", "Mean annual Precipitation (mm/year)", "Blues"),
    ]
    temp_columns = ("TemMin_climatology", "TemMax_climatology", "TemAver_climatology")
    temp_vmin = min(float(climatology[column].min()) for column in temp_columns)
    temp_vmax = max(float(climatology[column].max()) for column in temp_columns)
    precip_vmin = float(climatology["Precipitation_climatology"].min())
    precip_vmax = float(climatology["Precipitation_climatology"].max())
    xmin, xmax, ymin, ymax = CONFIG.figures.map_extent

    fig, axes = plt.subplots(2, 2, figsize=MANUSCRIPT_FIGURE_SIZES["regional_climatology"], constrained_layout=True)
    for ax, (column, title, cmap) in zip(axes.flat, panels):
        if column in temp_columns:
            vmin, vmax = temp_vmin, temp_vmax
        else:
            vmin, vmax = precip_vmin, precip_vmax

        china.boundary.plot(ax=ax, color="black", linewidth=0.6)
        provinces.boundary.plot(ax=ax, color="#999999", linewidth=0.3)
        scatter = ax.scatter(
            climatology["lon"],
            climatology["lat"],
            c=climatology[column],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker="s",
            s=16,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        _add_map_orientation(ax)
        fig.colorbar(scatter, ax=ax, fraction=0.046)

    return _save(fig, "regional_weather_climatology.png", output_dir)


def build_figures(run_id: str | None = None) -> list[Path]:
    run_paths = require_run(run_id=run_id)
    run_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        # framework(run_paths.figures_dir),
        study_area(run_paths.figures_dir),
        # evaluation_protocols(run_paths.figures_dir),
        # model_design_ladder(run_paths.figures_dir),
        overall_performance_summary(run_paths.figures_dir, run_paths.eval_dir),
        stage_specific_mae_decomposition(run_paths.figures_dir, run_paths.eval_dir),
        stage_bias_evolution(run_paths.figures_dir, run_paths.eval_dir),
    ]
    if _modifier_interpretability_samples_exist(run_paths.eval_dir):
        paths.append(modifier_interpretability(run_paths.figures_dir, run_paths.eval_dir))
    if _modifier_interpretability_samples_exist(run_paths.eval_dir, stages=SUPPLEMENTAL_MODIFIER_INTERPRETABILITY_STAGES):
        paths.append(supplemental_modifier_interpretability_early_stages(run_paths.figures_dir, run_paths.eval_dir))
    paths.extend(
        [
            regional_heading_maturity_comparison(run_paths.figures_dir, run_paths.eval_dir),
            regional_residual_diagnostics(run_paths.figures_dir, run_paths.eval_dir),
            regional_supplemental_model_heading_maturity_comparison(run_paths.figures_dir, run_paths.eval_dir),
            regional_supplemental_model_residual_diagnostics(run_paths.figures_dir, run_paths.eval_dir),
            # sample_distributions(run_paths.figures_dir),
            temporal_coverage_by_site_year(run_paths.figures_dir),
            # threshold_heterogeneity(run_paths.figures_dir),
            obs_vs_simulated(run_paths.figures_dir, run_paths.eval_dir),
            regional_weather_climatology(run_paths.figures_dir),
            regional_paired_scatter_diagnostics(run_paths.figures_dir, run_paths.eval_dir),
        ]
    )
    regional_sensitivity_metrics_path = _regional_reviving_offset_sensitivity_metrics_path()
    if regional_sensitivity_metrics_path.exists():
        paths.append(
            regional_reviving_offset_sensitivity(
                run_paths.figures_dir,
                metrics_path=regional_sensitivity_metrics_path,
            )
        )
    return paths
