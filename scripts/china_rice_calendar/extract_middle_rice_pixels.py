#!/usr/bin/env python3
"""Extract latitude-clipped Middle-rice pixel tables from ChinaRiceCalendar."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "external" / "china_rice_calendar" / "dataverse_v8" / "rice_pixels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts" / "features" / "china_rice_calendar"
OUTPUT_COLUMNS = ["lon", "lat", "transplanting_doy", "heading_doy", "maturity_doy"]
PERIODS = ("2003_2007", "2008_2012", "2013_2017", "2018_2022", "2003_2022")
DEFAULT_PERIOD = "2003_2022"
DEFAULT_OUTPUT_STEM = "middle_rice_pixels_lat16_35"
RASTER_STAGES = {
    "transplanting_doy": "transplanting",
    "heading_doy": "heading",
    "maturity_doy": "maturity",
}


def raster_files_for_period(period: str) -> dict[str, str]:
    if period not in PERIODS:
        raise ValueError(f"Unknown period {period!r}; expected one of {', '.join(PERIODS)}")
    return {
        column: f"Middle_rice_{stage}_dates_{period}_rice_pixels.tif"
        for column, stage in RASTER_STAGES.items()
    }


RASTER_FILES = raster_files_for_period(DEFAULT_PERIOD)


def resolve_period_data_dir(data_dir: Path, period: str) -> Path:
    """Return data_dir/period when present, otherwise allow data_dir to be a period directory."""
    period_dir = data_dir / period
    if period_dir.is_dir():
        return period_dir
    return data_dir


def missing_raster_paths(data_dir: Path, period: str) -> list[Path]:
    period_data_dir = resolve_period_data_dir(data_dir, period)
    return [
        period_data_dir / filename
        for filename in raster_files_for_period(period).values()
        if not (period_data_dir / filename).exists()
    ]


def available_periods(data_dir: Path) -> list[str]:
    return [period for period in PERIODS if not missing_raster_paths(data_dir, period)]


def _load_geotiff(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    image = Image.open(path)
    array = np.array(image, dtype=np.float32)
    scale = image.tag_v2.get(33550)
    tiepoint = image.tag_v2.get(33922)
    nodata_tag = image.tag_v2.get(42113)

    if scale is None or tiepoint is None:
        raise ValueError(f"Missing GeoTIFF georeferencing tags in {path}")

    nodata = float(nodata_tag) if nodata_tag is not None else np.nan
    return array, {
        "x0": float(tiepoint[3]),
        "y0": float(tiepoint[4]),
        "dx": float(scale[0]),
        "dy": float(scale[1]),
        "nodata": nodata,
    }


def _assert_compatible(reference: tuple[int, int], reference_meta: dict[str, float], name: str, array: np.ndarray, meta: dict[str, float]) -> None:
    if array.shape != reference:
        raise ValueError(f"Raster {name} shape mismatch: {array.shape} != {reference}")

    for key in ("x0", "y0", "dx", "dy"):
        if not np.isclose(meta[key], reference_meta[key]):
            raise ValueError(f"Raster {name} georeferencing mismatch on {key}: {meta[key]} != {reference_meta[key]}")


def _valid_mask(array: np.ndarray, nodata: float) -> np.ndarray:
    mask = np.isfinite(array)
    if np.isfinite(nodata):
        if abs(nodata) > 1e20:
            mask &= array > -1e20
        else:
            mask &= ~np.isclose(array, nodata)
    return mask


def build_middle_rice_pixel_table(
    data_dir: Path,
    *,
    period: str = DEFAULT_PERIOD,
    lat_min: float = 16.0,
    lat_max: float = 35.0,
) -> pd.DataFrame:
    if lat_min > lat_max:
        raise ValueError(f"lat_min must be <= lat_max, got {lat_min} > {lat_max}")

    arrays: dict[str, np.ndarray] = {}
    metas: dict[str, dict[str, float]] = {}
    raster_files = raster_files_for_period(period)
    period_data_dir = resolve_period_data_dir(data_dir, period)

    for column, filename in raster_files.items():
        path = period_data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required raster not found: {path}")
        array, meta = _load_geotiff(path)
        arrays[column] = array
        metas[column] = meta

    reference_column = OUTPUT_COLUMNS[2]
    reference_array = arrays[reference_column]
    reference_meta = metas[reference_column]

    for column in ("heading_doy", "maturity_doy"):
        _assert_compatible(reference_array.shape, reference_meta, column, arrays[column], metas[column])

    height, width = reference_array.shape
    lon = reference_meta["x0"] + (np.arange(width, dtype=np.float64) + 0.5) * reference_meta["dx"]
    lat = reference_meta["y0"] - (np.arange(height, dtype=np.float64) + 0.5) * abs(reference_meta["dy"])

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    valid = np.broadcast_to(lat_mask[:, None], (height, width)).copy()
    for column in ("transplanting_doy", "heading_doy", "maturity_doy"):
        valid &= _valid_mask(arrays[column], metas[column]["nodata"])

    rows, cols = np.where(valid)
    table = pd.DataFrame(
        {
            "lon": lon[cols],
            "lat": lat[rows],
            "transplanting_doy": arrays["transplanting_doy"][rows, cols].astype(np.float64),
            "heading_doy": arrays["heading_doy"][rows, cols].astype(np.float64),
            "maturity_doy": arrays["maturity_doy"][rows, cols].astype(np.float64),
        },
        columns=OUTPUT_COLUMNS,
    )
    return table.reset_index(drop=True)


def save_middle_rice_pixel_table(table: pd.DataFrame, output_dir: Path, output_stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_stem}.csv"
    parquet_path = output_dir / f"{output_stem}.parquet"
    table.to_csv(csv_path, index=False)
    table.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def period_output_dir(output_dir: Path, period: str, *, multiple_periods: bool) -> Path:
    if multiple_periods:
        return output_dir / period
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a latitude-clipped Middle-rice pixel table from ChinaRiceCalendar GeoTIFFs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"Directory containing period subdirectories or a single period of Middle-rice GeoTIFFs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Directory for derived table outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM, help="Output filename stem without extension.")
    parser.add_argument("--periods", nargs="+", choices=PERIODS, default=None, help="Periods to process. Defaults to all complete periods found under --data-dir.")
    parser.add_argument("--lat-min", type=float, default=16.0, help="Minimum latitude of pixel centers to retain.")
    parser.add_argument("--lat-max", type=float, default=35.0, help="Maximum latitude of pixel centers to retain.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lat_min > args.lat_max:
        parser.error("--lat-min must be less than or equal to --lat-max")

    periods = args.periods if args.periods is not None else available_periods(args.data_dir)
    if not periods:
        parser.error(f"No complete Middle-rice periods found under {args.data_dir}")

    multiple_periods = len(periods) > 1
    for period in periods:
        table = build_middle_rice_pixel_table(
            args.data_dir,
            period=period,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
        )
        output_dir = period_output_dir(args.output_dir, period, multiple_periods=multiple_periods)
        csv_path, parquet_path = save_middle_rice_pixel_table(table, output_dir, args.output_stem)

        print(f"Extracted {len(table):,} Middle-rice pixels for {period}")
        print(f"Latitude clip: {args.lat_min} to {args.lat_max}")
        print(f"CSV output: {csv_path}")
        print(f"Parquet output: {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
