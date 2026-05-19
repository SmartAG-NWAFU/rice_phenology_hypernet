#!/usr/bin/env python3
"""Coarsen Middle-rice ChinaRiceCalendar rasters with block-median aggregation."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "external" / "china_rice_calendar" / "dataverse_v8" / "rice_pixels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts" / "features" / "china_rice_calendar" / "middle_rice_0p05deg_median_min10_rasters"
PERIODS = ("2003_2007", "2008_2012", "2013_2017", "2018_2022", "2003_2022")
DEFAULT_PERIOD = "2003_2022"
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


def _write_geotiff(path: Path, array: np.ndarray, *, x0: float, y0: float, dx: float, dy: float, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(dx), float(dy), 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, float(x0), float(y0), 0.0)
    if np.isfinite(nodata):
        ifd[42113] = repr(float(nodata))
    Image.fromarray(array.astype(np.float32)).save(path, tiffinfo=ifd)


def _assert_compatible(reference_shape: tuple[int, int], reference_meta: dict[str, float], name: str, array: np.ndarray, meta: dict[str, float]) -> None:
    if array.shape != reference_shape:
        raise ValueError(f"Raster {name} shape mismatch: {array.shape} != {reference_shape}")
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


def _pad_to_block_size(array: np.ndarray, *, block_size: int, fill_value: float) -> np.ndarray:
    height, width = array.shape
    pad_h = (-height) % block_size
    pad_w = (-width) % block_size
    if pad_h == 0 and pad_w == 0:
        return array
    return np.pad(array, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=fill_value)


def _reshape_blocks(array: np.ndarray, block_size: int) -> np.ndarray:
    out_h = array.shape[0] // block_size
    out_w = array.shape[1] // block_size
    return array.reshape(out_h, block_size, out_w, block_size)


def coarsen_middle_rice_rasters(
    data_dir: Path,
    output_dir: Path,
    *,
    period: str = DEFAULT_PERIOD,
    block_size: int = 5,
    min_support: int = 10,
) -> dict[str, object]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if min_support <= 0:
        raise ValueError(f"min_support must be positive, got {min_support}")

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

    reference_array = arrays["transplanting_doy"]
    reference_meta = metas["transplanting_doy"]
    for column in ("heading_doy", "maturity_doy"):
        _assert_compatible(reference_array.shape, reference_meta, column, arrays[column], metas[column])

    triplet_valid = np.ones(reference_array.shape, dtype=bool)
    for column in ("transplanting_doy", "heading_doy", "maturity_doy"):
        triplet_valid &= _valid_mask(arrays[column], metas[column]["nodata"])

    padded_valid = _pad_to_block_size(triplet_valid.astype(np.int16), block_size=block_size, fill_value=0)
    support = _reshape_blocks(padded_valid, block_size).sum(axis=(1, 3)).astype(np.int32)
    coarse_valid = support >= min_support

    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_shape = tuple(int(v) for v in support.shape)
    valid_cells = int(coarse_valid.sum())

    for column, filename in raster_files.items():
        nodata = metas[column]["nodata"]
        padded_array = _pad_to_block_size(arrays[column], block_size=block_size, fill_value=nodata)
        padded_triplet = _pad_to_block_size(triplet_valid.astype(bool), block_size=block_size, fill_value=False)
        masked = np.where(padded_triplet, padded_array, np.nan)
        block_view = _reshape_blocks(masked, block_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            coarse = np.nanmedian(block_view, axis=(1, 3)).astype(np.float32)
        coarse[~coarse_valid] = np.float32(nodata)
        _write_geotiff(
            output_dir / filename,
            coarse,
            x0=reference_meta["x0"],
            y0=reference_meta["y0"],
            dx=reference_meta["dx"] * block_size,
            dy=reference_meta["dy"] * block_size,
            nodata=nodata,
        )

    return {
        "block_size": int(block_size),
        "min_support": int(min_support),
        "period": period,
        "coarse_shape": coarse_shape,
        "valid_cells": valid_cells,
        "output_dir": output_dir,
    }


def period_output_dir(output_dir: Path, period: str, *, multiple_periods: bool) -> Path:
    if multiple_periods:
        return output_dir / period
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coarsen Middle-rice ChinaRiceCalendar rasters using block-wise median aggregation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"Directory containing period subdirectories or a single period of Middle-rice GeoTIFFs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Directory for the coarsened GeoTIFFs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--periods", nargs="+", choices=PERIODS, default=None, help="Periods to process. Defaults to all complete periods found under --data-dir.")
    parser.add_argument("--block-size", type=int, default=5, help="Source-grid block width/height for each coarse cell.")
    parser.add_argument("--min-support", type=int, default=10, help="Minimum number of triplet-valid fine pixels required to keep a coarse cell.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.min_support <= 0:
        parser.error("--min-support must be positive")

    periods = args.periods if args.periods is not None else available_periods(args.data_dir)
    if not periods:
        parser.error(f"No complete Middle-rice periods found under {args.data_dir}")

    multiple_periods = len(periods) > 1
    for period in periods:
        output_dir = period_output_dir(args.output_dir, period, multiple_periods=multiple_periods)
        summary = coarsen_middle_rice_rasters(
            data_dir=args.data_dir,
            output_dir=output_dir,
            period=period,
            block_size=args.block_size,
            min_support=args.min_support,
        )
        print(f"Period: {summary['period']}")
        print(f"Coarse shape: {summary['coarse_shape'][0]} rows x {summary['coarse_shape'][1]} cols")
        print(f"Valid coarse cells: {summary['valid_cells']:,}")
        print(f"Block size: {summary['block_size']} fine pixels")
        print(f"Minimum support: {summary['min_support']} fine pixels")
        print(f"Output directory: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
