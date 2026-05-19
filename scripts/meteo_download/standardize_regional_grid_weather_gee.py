#!/usr/bin/env python3
"""Standardize regional GEE/ERA5-Land point weather and build climatology products."""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_DIR = DATA_DIR / "processed" / "regional_grid_weather_gee_era5_2003_2022"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed" / "regional_grid_weather_gee_era5_2003_2022_clean"

RAW_TO_STANDARD_COLUMNS = {
    "temperature_2m": "TemAver",
    "temperature_2m_min": "TemMin",
    "temperature_2m_max": "TemMax",
    "total_precipitation_sum": "Precipitation",
    "surface_solar_radiation_downwards_sum": "Radiation",
    "date": "Date",
}
RAW_REQUIRED_COLUMNS = ["point_id", "lon", "lat", *RAW_TO_STANDARD_COLUMNS.keys()]
DAILY_OUTPUT_COLUMNS = [
    "point_id",
    "lon",
    "lat",
    "Date",
    "year",
    "TemAver",
    "TemMin",
    "TemMax",
    "Precipitation",
    "Radiation",
]
POINT_YEAR_COLUMNS = [
    "point_id",
    "lon",
    "lat",
    "year",
    "n_days",
    "expected_days",
    "TemMin_year",
    "TemMax_year",
    "TemAver_year",
    "Precipitation_year",
    "Radiation_year",
]
CLIMATOLOGY_COLUMNS = [
    "point_id",
    "lon",
    "lat",
    "n_years_available",
    "TemMin_climatology",
    "TemMax_climatology",
    "TemAver_climatology",
    "Precipitation_climatology",
    "Radiation_climatology",
]

KELVIN_OFFSET = 273.15
PRECIPITATION_SCALE = 1000.0
RADIATION_SCALE = 1.0 / 1_000_000.0

DAILY_SCHEMA = pa.schema(
    [
        ("point_id", pa.string()),
        ("lon", pa.float64()),
        ("lat", pa.float64()),
        ("Date", pa.timestamp("ns")),
        ("year", pa.int32()),
        ("TemAver", pa.float64()),
        ("TemMin", pa.float64()),
        ("TemMax", pa.float64()),
        ("Precipitation", pa.float64()),
        ("Radiation", pa.float64()),
    ]
)


class QCAccumulator:
    """Mutable container for QC counters and ranges."""

    def __init__(self) -> None:
        self.total_point_files_found = 0
        self.processed_point_files = 0
        self.excluded_failed_points = 0
        self.total_rows = 0
        self.clipped_negative_precipitation_values = 0
        self.clipped_negative_radiation_values = 0
        self.variable_ranges = {
            column: {"min": None, "max": None}
            for column in ["TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]
        }
        self.per_shard: dict[str, dict[str, int]] = {}

    def update_ranges(self, df: pd.DataFrame) -> None:
        for column in self.variable_ranges:
            if column not in df.columns or df.empty:
                continue
            current_min = float(df[column].min())
            current_max = float(df[column].max())
            stored = self.variable_ranges[column]
            stored["min"] = current_min if stored["min"] is None else min(float(stored["min"]), current_min)
            stored["max"] = current_max if stored["max"] is None else max(float(stored["max"]), current_max)


def load_failed_point_ids(shard_dir: Path) -> set[str]:
    """Load the residual failed point ids for one shard."""
    failures_path = shard_dir / "failed_points.csv"
    if not failures_path.exists():
        return set()
    failures = pd.read_csv(failures_path)
    if "point_id" not in failures.columns or failures.empty:
        return set()
    return set(failures["point_id"].astype(str))


def _expected_days_in_year(year: int) -> int:
    return 366 if calendar.isleap(int(year)) else 365


def _empty_daily_table() -> pa.Table:
    return pa.Table.from_pylist([], schema=DAILY_SCHEMA)


def convert_raw_point_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert one raw GEE point frame into project-standard units and columns."""
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw point frame missing required columns: {missing}")

    converted = raw.loc[:, RAW_REQUIRED_COLUMNS].rename(columns=RAW_TO_STANDARD_COLUMNS).copy()
    converted["Date"] = pd.to_datetime(converted["Date"].astype(str), format="%Y%m%d", errors="raise")
    converted["year"] = converted["Date"].dt.year.astype("int32")

    for column in ["lon", "lat", "TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    converted = converted.dropna(subset=["point_id", "lon", "lat", "Date", "TemAver"]).copy()

    converted["TemAver"] = converted["TemAver"] - KELVIN_OFFSET
    converted["TemMin"] = converted["TemMin"] - KELVIN_OFFSET
    converted["TemMax"] = converted["TemMax"] - KELVIN_OFFSET
    converted["Precipitation"] = converted["Precipitation"] * PRECIPITATION_SCALE
    converted["Radiation"] = converted["Radiation"] * RADIATION_SCALE

    clipped_precip = int((converted["Precipitation"] < 0).sum())
    clipped_radiation = int((converted["Radiation"] < 0).sum())
    if clipped_precip > 0:
        converted["Precipitation"] = converted["Precipitation"].clip(lower=0)
    if clipped_radiation > 0:
        converted["Radiation"] = converted["Radiation"].clip(lower=0)

    converted = converted.loc[:, DAILY_OUTPUT_COLUMNS].sort_values("Date").reset_index(drop=True)
    return converted, {
        "clipped_negative_precipitation_values": clipped_precip,
        "clipped_negative_radiation_values": clipped_radiation,
    }


def summarize_point_year(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one standardized point frame to point-year metrics."""
    if clean_df.empty:
        return pd.DataFrame(columns=POINT_YEAR_COLUMNS)

    rows: list[dict[str, Any]] = []
    point_id = str(clean_df["point_id"].iloc[0])
    lon = float(clean_df["lon"].iloc[0])
    lat = float(clean_df["lat"].iloc[0])
    for year, group in clean_df.groupby("year", sort=True):
        year_int = int(year)
        rows.append(
            {
                "point_id": point_id,
                "lon": lon,
                "lat": lat,
                "year": year_int,
                "n_days": int(len(group)),
                "expected_days": _expected_days_in_year(year_int),
                "TemMin_year": float(group["TemMin"].mean()),
                "TemMax_year": float(group["TemMax"].mean()),
                "TemAver_year": float(group["TemAver"].mean()),
                "Precipitation_year": float(group["Precipitation"].sum()),
                "Radiation_year": float(group["Radiation"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=POINT_YEAR_COLUMNS)


def summarize_climatology(point_year_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse point-year summaries into 2003-2022 mean annual climatology."""
    if point_year_df.empty:
        return pd.DataFrame(columns=CLIMATOLOGY_COLUMNS)

    climatology = (
        point_year_df.groupby(["point_id", "lon", "lat"], as_index=False)
        .agg(
            n_years_available=("year", "nunique"),
            TemMin_climatology=("TemMin_year", "mean"),
            TemMax_climatology=("TemMax_year", "mean"),
            TemAver_climatology=("TemAver_year", "mean"),
            Precipitation_climatology=("Precipitation_year", "mean"),
            Radiation_climatology=("Radiation_year", "mean"),
        )
        .loc[:, CLIMATOLOGY_COLUMNS]
        .sort_values(["lat", "lon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return climatology


def _write_qc_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    json_path = output_dir / "regional_weather_qc_summary.json"
    csv_path = output_dir / "regional_weather_qc_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(csv_path, index=False)


def _iter_point_files(points_dir: Path) -> list[Path]:
    return sorted(points_dir.glob("*.csv"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize regional GEE/ERA5-Land point weather into clean daily and climatology products."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Root directory containing shard_*/points raw files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for standardized products (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(input_dir.glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under {input_dir}")

    qc = QCAccumulator()
    point_year_frames: list[pd.DataFrame] = []

    for shard_dir in shard_dirs:
        shard_name = shard_dir.name
        failed_point_ids = load_failed_point_ids(shard_dir)
        qc.excluded_failed_points += len(failed_point_ids)

        point_files = _iter_point_files(shard_dir / "points")
        qc.total_point_files_found += len(point_files)
        daily_output_path = output_dir / f"regional_weather_daily_clean_{shard_name}.parquet"
        writer = pq.ParquetWriter(daily_output_path, DAILY_SCHEMA)
        shard_rows = 0
        shard_points = 0

        try:
            for point_file in point_files:
                point_id = point_file.stem
                if point_id in failed_point_ids:
                    continue

                raw = pd.read_csv(point_file)
                clean_df, point_qc = convert_raw_point_frame(raw)
                table = pa.Table.from_pandas(clean_df, schema=DAILY_SCHEMA, preserve_index=False)
                writer.write_table(table)

                qc.processed_point_files += 1
                qc.total_rows += len(clean_df)
                shard_rows += len(clean_df)
                shard_points += 1
                qc.clipped_negative_precipitation_values += point_qc["clipped_negative_precipitation_values"]
                qc.clipped_negative_radiation_values += point_qc["clipped_negative_radiation_values"]
                qc.update_ranges(clean_df)
                point_year_frames.append(summarize_point_year(clean_df))

            if shard_points == 0:
                writer.write_table(_empty_daily_table())
        finally:
            writer.close()

        qc.per_shard[shard_name] = {
            "processed_point_files": shard_points,
            "total_rows": shard_rows,
            "failed_points_excluded": len(failed_point_ids),
        }

    point_year_df = (
        pd.concat(point_year_frames, ignore_index=True)
        if point_year_frames
        else pd.DataFrame(columns=POINT_YEAR_COLUMNS)
    )
    climatology_df = summarize_climatology(point_year_df)

    point_year_parquet = output_dir / "regional_weather_point_year_summary.parquet"
    point_year_csv = output_dir / "regional_weather_point_year_summary.csv"
    climatology_parquet = output_dir / "regional_weather_climatology_2003_2022.parquet"
    climatology_csv = output_dir / "regional_weather_climatology_2003_2022.csv"
    point_year_df.to_parquet(point_year_parquet, index=False)
    point_year_df.to_csv(point_year_csv, index=False)
    climatology_df.to_parquet(climatology_parquet, index=False)
    climatology_df.to_csv(climatology_csv, index=False)

    year_coverage: dict[str, dict[str, int]] = {}
    if not point_year_df.empty:
        for year, group in point_year_df.groupby("year", sort=True):
            year_coverage[str(int(year))] = {
                "point_years": int(len(group)),
                "incomplete_point_years": int((group["n_days"] != group["expected_days"]).sum()),
            }

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_point_files_found": qc.total_point_files_found,
        "processed_point_files": qc.processed_point_files,
        "excluded_failed_points": qc.excluded_failed_points,
        "total_rows": qc.total_rows,
        "clipped_negative_precipitation_values": qc.clipped_negative_precipitation_values,
        "clipped_negative_radiation_values": qc.clipped_negative_radiation_values,
        "variable_ranges": qc.variable_ranges,
        "year_coverage": year_coverage,
        "per_shard": qc.per_shard,
        "point_year_output": str(point_year_parquet),
        "climatology_output": str(climatology_parquet),
    }
    _write_qc_summary(output_dir, summary)

    print(f"Processed point files: {qc.processed_point_files}")
    print(f"Excluded failed points: {qc.excluded_failed_points}")
    print(f"Daily clean output dir: {output_dir}")
    print(f"Point-year summary: {point_year_parquet}")
    print(f"Climatology summary: {climatology_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
