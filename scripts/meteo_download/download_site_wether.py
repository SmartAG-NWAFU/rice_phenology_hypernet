#!/usr/bin/env python3
"""Download historical station weather via local GEE/ERA5-Land."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
HOST_OUTPUT_MOUNT_ROOT = DATA_DIR / "processed"
CONTAINER_OUTPUT_MOUNT_ROOT = Path("/app/outputs")

DEFAULT_CATALOG_PATH = DATA_DIR / "raw" / "obser_pheno_catalog_origin.xlsx"
DEFAULT_OUTPUT_PATH = DATA_DIR / "raw" / "daily_temperature.csv"
DEFAULT_SCRATCH_DIR = DATA_DIR / "processed" / "site_weather_gee_era5_1981_2020_raw"
DEFAULT_START_DATE = "1981-01-01"
DEFAULT_END_DATE = "2020-12-31"
DEFAULT_BATCH_SIZE = 12
DEFAULT_MAX_WORKERS = 6
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 3600

GEE_EXTRACT_VARS = [
    "temperature_2m",
    "temperature_2m_min",
    "temperature_2m_max",
    "total_precipitation_sum",
    "surface_solar_radiation_downwards_sum",
]
RAW_OUTPUT_COLUMNS = [*GEE_EXTRACT_VARS, "date"]
OUTPUT_COLUMNS = ["SID", "year", "Date", "TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]
FAILURE_COLUMNS = ["station_index", "SID", "lon", "lat", "batch_index", "error"]

RAW_TO_STANDARD_COLUMNS = {
    "temperature_2m": "TemAver",
    "temperature_2m_min": "TemMin",
    "temperature_2m_max": "TemMax",
    "total_precipitation_sum": "Precipitation",
    "surface_solar_radiation_downwards_sum": "Radiation",
    "date": "Date",
}
KELVIN_OFFSET = 273.15
PRECIPITATION_SCALE = 1000.0
RADIATION_SCALE = 1.0 / 1_000_000.0


def prepare_stations(catalog_path: Path) -> pd.DataFrame:
    """Load and normalize the station catalog."""
    stations = pd.read_excel(catalog_path)
    if "station ID" in stations.columns:
        stations = stations.rename(columns={"station ID": "SID"})

    required = {"SID", "lat", "lon"}
    missing = sorted(required - set(stations.columns))
    if missing:
        raise ValueError(f"Missing required columns in catalog: {missing}")

    stations = stations.loc[:, ["SID", "lat", "lon"]].copy()
    stations["SID"] = stations["SID"].astype(str)
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations = stations.dropna(subset=["SID", "lat", "lon"]).copy()
    stations = stations.drop_duplicates(subset=["SID", "lat", "lon"]).reset_index(drop=True)

    duplicate_sids = stations[stations.duplicated(subset=["SID"], keep=False)]
    if not duplicate_sids.empty:
        mismatched = duplicate_sids.groupby("SID")[["lat", "lon"]].nunique()
        mismatched = mismatched[(mismatched["lat"] > 1) | (mismatched["lon"] > 1)]
        if not mismatched.empty:
            preview = ", ".join(mismatched.index.astype(str)[:5])
            suffix = "..." if len(mismatched) > 5 else ""
            print(
                "Warning: multiple lat/lon entries for some stations; "
                f"keeping the first occurrence (e.g. {preview}{suffix})."
            )
        stations = stations.sort_values("SID").drop_duplicates(subset=["SID"], keep="first").reset_index(drop=True)

    stations = stations.sort_values("SID").reset_index(drop=True)
    stations["station_index"] = stations.index.astype(int)
    return stations.loc[:, ["station_index", "SID", "lon", "lat"]]


def expected_date_strings(start_date: str, end_date: str) -> list[str]:
    """Build the expected daily date sequence."""
    return pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()


def gee_exclusive_end_date(end_date: str) -> str:
    """Translate the CLI inclusive end date into the GEE service's exclusive upper bound."""
    return (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def host_to_container_path(host_path: Path) -> Path:
    """Translate a host path under data/processed to the container output mount."""
    resolved_host = host_path.resolve()
    resolved_root = HOST_OUTPUT_MOUNT_ROOT.resolve()
    try:
        relative = resolved_host.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Scratch path must be under {resolved_root}, got {resolved_host}"
        ) from exc
    return CONTAINER_OUTPUT_MOUNT_ROOT / relative


def check_health(*, base_url: str, timeout_seconds: int) -> dict[str, object]:
    """Validate the local meteo service before downloading."""
    endpoint = f"{base_url.rstrip('/')}/health"
    request = Request(endpoint, method="GET")

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Health check failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Health check request error: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode health check response: {raw[:500]}") from exc

    if status_code != 200 or not bool(payload.get("ok", False)):
        raise RuntimeError(f"Local meteo service unhealthy: {payload}")
    return payload


def build_batch_payload(
    batch_stations: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    container_batch_dir: Path,
    max_workers: int,
) -> dict[str, object]:
    """Build one /gee/meteo request payload for a station batch."""
    return {
        "points": [
            {
                "id": str(station["SID"]),
                "latitude": float(station["lat"]),
                "longitude": float(station["lon"]),
            }
            for _, station in batch_stations.iterrows()
        ],
        "time_range": [start_date, gee_exclusive_end_date(end_date)],
        "extract_vars": GEE_EXTRACT_VARS,
        "output_dir": str(container_batch_dir),
        "data_id": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "max_workers": int(max_workers),
        "authenticate": False,
    }


def query_gee_meteo(
    *,
    base_url: str,
    payload: dict[str, object],
    request_timeout_seconds: int,
) -> dict[str, object]:
    """Call the local /gee/meteo endpoint and return the parsed JSON payload."""
    endpoint = f"{base_url.rstrip('/')}/gee/meteo"
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=request_timeout_seconds) as response:
            status_code = response.getcode()
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GEE meteo request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"GEE meteo request error: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode GEE meteo response: {raw[:500]}") from exc

    if status_code != 200 or not bool(result.get("ok", False)):
        raise RuntimeError(f"GEE meteo returned non-success payload: {result}")
    return result


def load_batch_manifest(batch_host_dir: Path) -> dict[str, object]:
    """Load the raw batch manifest produced by the meteo service."""
    manifest_path = batch_host_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing batch manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def convert_raw_station_frame(raw: pd.DataFrame, *, sid: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert one raw GEE station frame into the project's daily weather schema."""
    missing = [column for column in RAW_OUTPUT_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw station frame missing required columns: {missing}")

    converted = raw.loc[:, RAW_OUTPUT_COLUMNS].rename(columns=RAW_TO_STANDARD_COLUMNS).copy()
    converted["Date"] = pd.to_datetime(converted["Date"].astype(str), format="%Y%m%d", errors="raise")
    converted["year"] = converted["Date"].dt.year.astype(int)
    converted.insert(0, "SID", str(sid))

    for column in ["TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")

    converted = converted.dropna(subset=["Date", "TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]).copy()
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

    converted["Date"] = converted["Date"].dt.strftime("%Y-%m-%d")
    converted = converted.loc[:, OUTPUT_COLUMNS].sort_values("Date").reset_index(drop=True)
    return converted, {
        "clipped_negative_precipitation_values": clipped_precip,
        "clipped_negative_radiation_values": clipped_radiation,
    }


def normalize_station_output(point_file: Path, *, sid: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load one raw point CSV and convert it into the final station schema."""
    try:
        raw = pd.read_csv(point_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Station output is empty: {point_file}") from exc
    return convert_raw_station_frame(raw, sid=sid)


def station_file_is_complete(path: Path, *, expected_dates: list[str]) -> bool:
    """Check whether one standardized station file fully covers the requested date range."""
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        data = pd.read_csv(path, dtype={"SID": str})
    except (pd.errors.EmptyDataError, ValueError):
        return False

    if list(data.columns) != OUTPUT_COLUMNS:
        return False
    if len(data) != len(expected_dates):
        return False
    if data[["SID", "Date"]].duplicated().any():
        return False
    return list(data["Date"].astype(str)) == expected_dates


def dedupe_failures(failures: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the most recent failure row for each station id."""
    latest_by_sid: dict[str, dict[str, object]] = {}
    for row in failures:
        latest_by_sid[str(row["SID"])] = {
            "station_index": int(row["station_index"]),
            "SID": str(row["SID"]),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "batch_index": int(row["batch_index"]),
            "error": str(row["error"]),
        }
    return sorted(latest_by_sid.values(), key=lambda row: int(row["station_index"]))


def iter_batches(stations: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Split a station frame into stable fixed-size batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [
        stations.iloc[start : start + batch_size].reset_index(drop=True)
        for start in range(0, len(stations), batch_size)
    ]


def write_summary(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON summary payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_batch(
    batch_stations: pd.DataFrame,
    *,
    batch_index: int,
    base_url: str,
    batch_host_dir: Path,
    batch_container_dir: Path,
    points_dir: Path,
    start_date: str,
    end_date: str,
    max_workers: int,
    request_timeout_seconds: int,
) -> dict[str, object]:
    """Run one GEE batch request and normalize the resulting station files."""
    if batch_host_dir.exists():
        shutil.rmtree(batch_host_dir)
    batch_host_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = build_batch_payload(
            batch_stations,
            start_date=start_date,
            end_date=end_date,
            container_batch_dir=batch_container_dir,
            max_workers=max_workers,
        )

        try:
            query_gee_meteo(
                base_url=base_url,
                payload=payload,
                request_timeout_seconds=request_timeout_seconds,
            )
            manifest = load_batch_manifest(batch_host_dir)
        except Exception as exc:
            failed_stations = [
                {
                    "station_index": int(station["station_index"]),
                    "SID": str(station["SID"]),
                    "lon": float(station["lon"]),
                    "lat": float(station["lat"]),
                    "batch_index": int(batch_index),
                    "error": str(exc),
                }
                for _, station in batch_stations.iterrows()
            ]
            return {
                "batch_failed": True,
                "completed_points": 0,
                "failed_points": failed_stations,
                "missing_output_points": len(batch_stations),
                "point_qc": {"clipped_negative_precipitation_values": 0, "clipped_negative_radiation_values": 0},
            }

        actual_file_map = {path.stem: path for path in batch_host_dir.glob("*.csv")}
        requested_ids = set(batch_stations["SID"].astype(str))
        completed_points = 0
        missing_output_points = 0
        failed_points: list[dict[str, object]] = []
        point_qc = {"clipped_negative_precipitation_values": 0, "clipped_negative_radiation_values": 0}

        points_dir.mkdir(parents=True, exist_ok=True)
        for _, station in batch_stations.iterrows():
            sid = str(station["SID"])
            source_file = actual_file_map.get(sid)
            if source_file is None:
                missing_output_points += 1
                failed_points.append(
                    {
                        "station_index": int(station["station_index"]),
                        "SID": sid,
                        "lon": float(station["lon"]),
                        "lat": float(station["lat"]),
                        "batch_index": int(batch_index),
                        "error": "Missing point output file in batch result",
                    }
                )
                continue

            try:
                normalized, qc = normalize_station_output(source_file, sid=sid)
                normalized.to_csv(points_dir / f"{sid}.csv", index=False)
                completed_points += 1
                point_qc["clipped_negative_precipitation_values"] += int(qc["clipped_negative_precipitation_values"])
                point_qc["clipped_negative_radiation_values"] += int(qc["clipped_negative_radiation_values"])
            except Exception as exc:
                failed_points.append(
                    {
                        "station_index": int(station["station_index"]),
                        "SID": sid,
                        "lon": float(station["lon"]),
                        "lat": float(station["lat"]),
                        "batch_index": int(batch_index),
                        "error": str(exc),
                    }
                )

        manifest_count = int(manifest.get("file_count", 0) or 0)
        actual_count = len(actual_file_map)
        if manifest_count < len(requested_ids) or actual_count < len(requested_ids):
            missing_output_points = max(missing_output_points, len(requested_ids) - completed_points)

        return {
            "batch_failed": False,
            "completed_points": completed_points,
            "failed_points": failed_points,
            "missing_output_points": missing_output_points,
            "point_qc": point_qc,
        }
    finally:
        if batch_host_dir.exists():
            shutil.rmtree(batch_host_dir, ignore_errors=True)


def retry_failures(
    failures: list[dict[str, object]],
    *,
    base_url: str,
    batches_dir: Path,
    points_dir: Path,
    start_date: str,
    end_date: str,
    request_timeout_seconds: int,
    batch_index_start: int,
) -> dict[str, object]:
    """Retry each failed station once as a single-point request."""
    final_failures: list[dict[str, object]] = []
    repair_completed_points = 0
    missing_output_points = 0
    point_qc = {"clipped_negative_precipitation_values": 0, "clipped_negative_radiation_values": 0}

    retry_rows = dedupe_failures(failures)
    if not retry_rows:
        return {
            "final_failures": [],
            "repair_attempted_points": 0,
            "repair_completed_points": 0,
            "missing_output_points": 0,
            "point_qc": point_qc,
        }

    retry_stations = pd.DataFrame(retry_rows)
    retry_stations = retry_stations.loc[:, ["station_index", "SID", "lon", "lat"]]

    for offset, batch_stations in enumerate(iter_batches(retry_stations, 1), start=0):
        batch_index = batch_index_start + offset
        sid = str(batch_stations["SID"].iloc[0])
        print(f"[retry {offset + 1}/{len(retry_stations)}] SID={sid} max_workers=1")
        batch_host_dir = batches_dir / f"retry_{batch_index:06d}"
        batch_container_dir = host_to_container_path(batch_host_dir)
        result = process_batch(
            batch_stations,
            batch_index=batch_index,
            base_url=base_url,
            batch_host_dir=batch_host_dir,
            batch_container_dir=batch_container_dir,
            points_dir=points_dir,
            start_date=start_date,
            end_date=end_date,
            max_workers=1,
            request_timeout_seconds=request_timeout_seconds,
        )
        repair_completed_points += int(result["completed_points"])
        missing_output_points += int(result["missing_output_points"])
        point_qc["clipped_negative_precipitation_values"] += int(result["point_qc"]["clipped_negative_precipitation_values"])
        point_qc["clipped_negative_radiation_values"] += int(result["point_qc"]["clipped_negative_radiation_values"])
        final_failures.extend(result["failed_points"])
        print(
            f"[retry {offset + 1}/{len(retry_stations)}] "
            f"SID={sid} completed={result['completed_points']} failed={len(result['failed_points'])}"
        )

    return {
        "final_failures": dedupe_failures(final_failures),
        "repair_attempted_points": len(retry_stations),
        "repair_completed_points": repair_completed_points,
        "missing_output_points": missing_output_points,
        "point_qc": point_qc,
    }


def aggregate_station_files(points_dir: Path, stations: pd.DataFrame) -> pd.DataFrame:
    """Concatenate standardized station files into one candidate frame."""
    frames: list[pd.DataFrame] = []
    for _, station in stations.iterrows():
        point_path = points_dir / f"{station['SID']}.csv"
        if not point_path.exists():
            continue
        frame = pd.read_csv(point_path, dtype={"SID": str})
        frames.append(frame.loc[:, OUTPUT_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce").fillna(0).astype(int)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    sid_sort = pd.to_numeric(combined["SID"], errors="coerce")
    combined = (
        combined.assign(_sid_sort=sid_sort)
        .drop_duplicates(subset=["SID", "Date"], keep="last")
        .sort_values(["_sid_sort", "SID", "Date"])
        .drop(columns="_sid_sort")
        .reset_index(drop=True)
    )
    return combined.loc[:, OUTPUT_COLUMNS]


def validate_candidate(
    candidate: pd.DataFrame,
    *,
    stations: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    """Validate that the candidate file fully covers the requested domain."""
    if list(candidate.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Candidate columns do not match required schema: {list(candidate.columns)}")

    expected_dates = expected_date_strings(start_date, end_date)
    expected_n_days = len(expected_dates)
    expected_sids = stations["SID"].astype(str).tolist()
    actual_sids = sorted(candidate["SID"].astype(str).unique().tolist())

    missing_sids = sorted(set(expected_sids) - set(actual_sids))
    extra_sids = sorted(set(actual_sids) - set(expected_sids))
    if missing_sids or extra_sids:
        raise ValueError(f"Station coverage mismatch. Missing={missing_sids} Extra={extra_sids}")

    if candidate[["SID", "Date"]].duplicated().any():
        duplicate_rows = candidate.loc[candidate[["SID", "Date"]].duplicated(), ["SID", "Date"]]
        raise ValueError(f"Candidate contains duplicated SID-Date rows: {duplicate_rows.head(10).to_dict('records')}")

    coverage_errors: list[str] = []
    for sid, group in candidate.groupby("SID", sort=True):
        actual_dates = group["Date"].astype(str).tolist()
        if len(actual_dates) != expected_n_days:
            coverage_errors.append(f"{sid}: expected {expected_n_days} rows, got {len(actual_dates)}")
            continue
        if actual_dates != expected_dates:
            missing = sorted(set(expected_dates) - set(actual_dates))
            preview = ",".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            coverage_errors.append(f"{sid}: missing {len(missing)} dates ({preview}{suffix})")

    expected_total_rows = len(expected_sids) * expected_n_days
    if len(candidate) != expected_total_rows:
        coverage_errors.append(
            f"total rows mismatch: expected {expected_total_rows}, got {len(candidate)}"
        )

    if coverage_errors:
        raise ValueError("; ".join(coverage_errors))

    return {
        "station_count": len(expected_sids),
        "expected_days_per_station": expected_n_days,
        "expected_total_rows": expected_total_rows,
        "actual_total_rows": int(len(candidate)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Download historical station weather from local GEE/ERA5-Land and write daily_temperature.csv."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help=f"Station catalog path (default: {DEFAULT_CATALOG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Final output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"End date in YYYY-MM-DD format (default: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing standardized station scratch files instead of resuming from complete ones.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Local meteo API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of stations per GEE batch request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"max_workers passed to /gee/meteo (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_SCRATCH_DIR,
        help=f"Scratch directory under data/processed (default: {DEFAULT_SCRATCH_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the station downloader."""
    parser = build_parser()
    args = parser.parse_args(argv)

    health = check_health(
        base_url=str(args.base_url),
        timeout_seconds=min(int(args.request_timeout_seconds), 30),
    )

    stations = prepare_stations(Path(args.catalog))
    expected_dates = expected_date_strings(str(args.start_date), str(args.end_date))

    scratch_dir = Path(args.scratch_dir)
    points_dir = scratch_dir / "points"
    batches_dir = scratch_dir / "batches"
    manifest_path = scratch_dir / "station_manifest.csv"
    summary_path = scratch_dir / "download_summary.json"
    failures_path = scratch_dir / "failed_points.csv"
    candidate_path = scratch_dir / "daily_temperature_candidate.csv"

    scratch_dir.mkdir(parents=True, exist_ok=True)
    stations.to_csv(manifest_path, index=False)

    todo_rows = []
    skipped_complete_points = 0
    for _, station in stations.iterrows():
        station_path = points_dir / f"{station['SID']}.csv"
        if not bool(args.overwrite) and station_file_is_complete(station_path, expected_dates=expected_dates):
            skipped_complete_points += 1
            continue
        todo_rows.append(station.to_dict())

    todo_stations = pd.DataFrame(todo_rows, columns=stations.columns)
    summary: dict[str, object] = {
        "catalog": str(args.catalog),
        "output": str(args.output),
        "scratch_dir": str(scratch_dir),
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "batch_size": int(args.batch_size),
        "max_workers": int(args.max_workers),
        "base_url": str(args.base_url),
        "health": health,
        "total_stations": int(len(stations)),
        "completed_points": 0,
        "skipped_complete_points": int(skipped_complete_points),
        "failed_points": 0,
        "completed_batches": 0,
        "failed_batches": 0,
        "missing_output_points": 0,
        "repair_attempted_points": 0,
        "repair_completed_points": 0,
        "repair_failed_points": 0,
        "manifest_path": str(manifest_path),
        "failures_path": str(failures_path),
        "candidate_path": str(candidate_path),
        "validation_passed": False,
        "output_written": False,
        "last_processed_sid": None,
        "clipped_negative_precipitation_values": 0,
        "clipped_negative_radiation_values": 0,
    }
    write_summary(summary_path, summary)
    pd.DataFrame(columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

    print(
        f"Starting station download: total={len(stations)} "
        f"todo={len(todo_stations)} skipped_complete={skipped_complete_points} "
        f"batch_size={args.batch_size} max_workers={args.max_workers}"
    )
    print(f"Scratch: {scratch_dir}")
    print(f"Summary: {summary_path}")

    failures: list[dict[str, object]] = []
    if not todo_stations.empty:
        batches = iter_batches(todo_stations, int(args.batch_size))
        for batch_idx, batch_stations in enumerate(batches, start=1):
            sid_preview = ",".join(batch_stations["SID"].astype(str).tolist())
            print(f"[batch {batch_idx}/{len(batches)}] requesting SID={sid_preview}")
            batch_host_dir = batches_dir / f"batch_{batch_idx:06d}"
            batch_container_dir = host_to_container_path(batch_host_dir)
            result = process_batch(
                batch_stations,
                batch_index=batch_idx,
                base_url=str(args.base_url),
                batch_host_dir=batch_host_dir,
                batch_container_dir=batch_container_dir,
                points_dir=points_dir,
                start_date=str(args.start_date),
                end_date=str(args.end_date),
                max_workers=int(args.max_workers),
                request_timeout_seconds=int(args.request_timeout_seconds),
            )

            summary["completed_points"] = int(summary["completed_points"]) + int(result["completed_points"])
            summary["missing_output_points"] = int(summary["missing_output_points"]) + int(result["missing_output_points"])
            summary["clipped_negative_precipitation_values"] = int(summary["clipped_negative_precipitation_values"]) + int(result["point_qc"]["clipped_negative_precipitation_values"])
            summary["clipped_negative_radiation_values"] = int(summary["clipped_negative_radiation_values"]) + int(result["point_qc"]["clipped_negative_radiation_values"])
            if bool(result["batch_failed"]):
                summary["failed_batches"] = int(summary["failed_batches"]) + 1
            else:
                summary["completed_batches"] = int(summary["completed_batches"]) + 1

            failures.extend(result["failed_points"])
            failures = dedupe_failures(failures)
            summary["failed_points"] = len(failures)
            if not batch_stations.empty:
                summary["last_processed_sid"] = str(batch_stations["SID"].iloc[-1])

            write_summary(summary_path, summary)
            pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)
            print(
                f"[batch {batch_idx}/{len(batches)}] "
                f"completed={result['completed_points']} failed={len(result['failed_points'])} "
                f"missing={result['missing_output_points']} last_sid={summary['last_processed_sid']}"
            )

        repair = retry_failures(
            failures,
            base_url=str(args.base_url),
            batches_dir=batches_dir,
            points_dir=points_dir,
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            request_timeout_seconds=int(args.request_timeout_seconds),
            batch_index_start=len(batches) + 1,
        )
        failures = dedupe_failures(repair["final_failures"])
        summary["completed_points"] = int(summary["completed_points"]) + int(repair["repair_completed_points"])
        summary["missing_output_points"] = int(summary["missing_output_points"]) + int(repair["missing_output_points"])
        summary["repair_attempted_points"] = int(repair["repair_attempted_points"])
        summary["repair_completed_points"] = int(repair["repair_completed_points"])
        summary["repair_failed_points"] = len(failures)
        summary["failed_points"] = len(failures)
        summary["clipped_negative_precipitation_values"] = int(summary["clipped_negative_precipitation_values"]) + int(repair["point_qc"]["clipped_negative_precipitation_values"])
        summary["clipped_negative_radiation_values"] = int(summary["clipped_negative_radiation_values"]) + int(repair["point_qc"]["clipped_negative_radiation_values"])
        write_summary(summary_path, summary)
        pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

    if batches_dir.exists():
        try:
            batches_dir.rmdir()
        except OSError:
            pass

    candidate = aggregate_station_files(points_dir, stations)
    candidate.to_csv(candidate_path, index=False)

    try:
        validation = validate_candidate(
            candidate,
            stations=stations,
            start_date=str(args.start_date),
            end_date=str(args.end_date),
        )
    except Exception as exc:
        summary["validation_passed"] = False
        summary["validation_error"] = str(exc)
        write_summary(summary_path, summary)
        print(f"Candidate: {candidate_path}")
        print(f"Summary: {summary_path}")
        print(f"Failures: {failures_path}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_path, index=False)

    summary["validation_passed"] = True
    summary["validation"] = validation
    summary["output_written"] = True
    write_summary(summary_path, summary)

    print(f"Candidate: {candidate_path}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Failures: {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
