#!/usr/bin/env python3
"""Download historical daily weather for regional coarse-grid points via local GEE/ERA5."""

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

DEFAULT_POINTS_PATH = (
    DATA_DIR
    / "artifacts"
    / "features"
    / "2003_2022"
    / "china_rice_calendar"
    / "2003_2022"
    / "middle_rice_pixels_0p05deg_median_min10_lat16_35.parquet"
)
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed" / "regional_grid_weather_gee_era5_2003_2022"
DEFAULT_START_DATE = "2003-01-01"
DEFAULT_END_DATE = "2022-12-31"
DEFAULT_NUM_SHARDS = 3
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
FINAL_OUTPUT_COLUMNS = ["point_id", "lon", "lat", *RAW_OUTPUT_COLUMNS]
FAILURE_COLUMNS = ["point_index", "point_id", "lon", "lat", "batch_index", "error"]

FAILURE_KIND_MISSING_OUTPUT = "missing_output_file"
FAILURE_KIND_TIMEOUT = "computation_timed_out"
FAILURE_KIND_NON_RETRYABLE = "non_retryable"


def make_point_id(lon: float, lat: float) -> str:
    """Build a stable filesystem-safe point identifier from lon/lat."""

    def _format(value: float) -> str:
        return f"{float(value):.6f}".replace("-", "m").replace(".", "p")

    return f"lon_{_format(lon)}_lat_{_format(lat)}"


def prepare_points(points_path: Path) -> pd.DataFrame:
    """Load and validate the point table, then assign stable point ids and indices."""
    suffix = points_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(points_path)
    elif suffix == ".csv":
        df = pd.read_csv(points_path)
    else:
        raise ValueError(f"Unsupported points file format: {points_path}")

    required = {"lon", "lat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required point columns: {missing}")

    points = df.loc[:, ["lon", "lat"]].copy()
    points["lon"] = pd.to_numeric(points["lon"], errors="coerce")
    points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
    points = points.dropna(subset=["lon", "lat"]).reset_index(drop=True)

    duplicate_count = int(points.duplicated(subset=["lon", "lat"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Point table still has duplicated lon/lat pairs: {duplicate_count}")

    points = points.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)
    points["point_index"] = points.index.astype(int)
    points["point_id"] = [make_point_id(lon, lat) for lon, lat in zip(points["lon"], points["lat"])]
    return points.loc[:, ["point_index", "point_id", "lon", "lat"]]


def select_points_for_shard(
    points: pd.DataFrame,
    *,
    num_shards: int,
    shard_index: int,
    max_points: int | None = None,
) -> pd.DataFrame:
    """Select the subset of points assigned to one shard."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}")

    shard_points = points.copy()
    shard_points["assigned_shard"] = shard_points["point_index"] % int(num_shards)
    shard_points = shard_points[shard_points["assigned_shard"] == int(shard_index)].reset_index(drop=True)
    if max_points is not None:
        shard_points = shard_points.head(int(max_points)).reset_index(drop=True)
    return shard_points.loc[:, ["point_index", "point_id", "lon", "lat", "assigned_shard"]]


def point_file_is_complete(path: Path) -> bool:
    """Treat any non-empty final point CSV as complete for this raw-download stage."""
    return path.exists() and path.is_file() and path.stat().st_size > 0


def classify_failure(error: str) -> str:
    """Collapse raw error text into the retry categories used by this downloader."""
    if "Missing point output file in batch result" in str(error):
        return FAILURE_KIND_MISSING_OUTPUT
    if 'Computation timed out.' in str(error):
        return FAILURE_KIND_TIMEOUT
    return FAILURE_KIND_NON_RETRYABLE


def dedupe_failures(failures: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep only the latest failure row for each point id."""
    if not failures:
        return []

    latest_by_point: dict[str, dict[str, object]] = {}
    for row in failures:
        latest_by_point[str(row["point_id"])] = {
            "point_index": int(row["point_index"]),
            "point_id": str(row["point_id"]),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "batch_index": int(row["batch_index"]),
            "error": str(row["error"]),
        }

    return sorted(latest_by_point.values(), key=lambda row: int(row["point_index"]))


def failure_rows_to_points_frame(failures: list[dict[str, object]]) -> pd.DataFrame:
    """Project failure rows back to the point frame required by process_batch."""
    if not failures:
        return pd.DataFrame(columns=["point_index", "point_id", "lon", "lat"])

    frame = pd.DataFrame(dedupe_failures(failures))
    return frame.loc[:, ["point_index", "point_id", "lon", "lat"]].sort_values(
        "point_index"
    ).reset_index(drop=True)


def host_to_container_path(host_path: Path) -> Path:
    """Translate a host path under data/processed into the container /app/outputs mount."""
    resolved_host = host_path.resolve()
    resolved_root = HOST_OUTPUT_MOUNT_ROOT.resolve()
    try:
        relative = resolved_host.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Output path must be under {resolved_root}, got {resolved_host}"
        ) from exc
    return CONTAINER_OUTPUT_MOUNT_ROOT / relative


def build_batch_payload(
    batch_points: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    container_batch_dir: Path,
    max_workers: int,
) -> dict[str, object]:
    """Build one /gee/meteo request payload for a point batch."""
    return {
        "points": [
            {
                "id": str(point["point_id"]),
                "latitude": float(point["lat"]),
                "longitude": float(point["lon"]),
            }
            for _, point in batch_points.iterrows()
        ],
        "time_range": [start_date, end_date],
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
    batch_host_dir: Path,
) -> dict[str, object]:
    """Call the local /gee/meteo endpoint and return the parsed JSON payload."""
    del batch_host_dir
    body = json.dumps(payload).encode("utf-8")
    endpoint = f"{base_url.rstrip('/')}/gee/meteo"
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
        raise RuntimeError(
            f"GEE meteo request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"GEE meteo request error: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode GEE meteo JSON response: {raw[:500]}") from exc

    if status_code != 200 or not result.get("ok", False):
        raise RuntimeError(f"GEE meteo returned non-success payload: {result}")
    return result


def load_batch_manifest(batch_host_dir: Path) -> dict[str, object]:
    """Load the batch manifest produced by the local GEE service."""
    manifest_path = batch_host_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing batch manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def normalize_point_output(point_file: Path, *, point_id: str, lon: float, lat: float) -> pd.DataFrame:
    """Validate one raw GEE point CSV and attach stable point metadata columns."""
    try:
        data = pd.read_csv(point_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Point output is empty: {point_file}") from exc

    missing = [col for col in RAW_OUTPUT_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Point output missing required columns {missing}: {point_file}")
    if data.empty:
        raise ValueError(f"Point output has no rows: {point_file}")

    out = data.loc[:, RAW_OUTPUT_COLUMNS].copy()
    out.insert(0, "lat", float(lat))
    out.insert(0, "lon", float(lon))
    out.insert(0, "point_id", str(point_id))
    return out.loc[:, FINAL_OUTPUT_COLUMNS]


def process_batch(
    batch_points: pd.DataFrame,
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
    """Run one GEE batch request, then normalize batch outputs into final point CSVs."""
    if batch_host_dir.exists():
        shutil.rmtree(batch_host_dir)
    batch_host_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_batch_payload(
            batch_points,
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
                batch_host_dir=batch_host_dir,
            )
            manifest = load_batch_manifest(batch_host_dir)
        except Exception as exc:
            failed_points = [
                {
                    "point_index": int(point["point_index"]),
                    "point_id": str(point["point_id"]),
                    "lon": float(point["lon"]),
                    "lat": float(point["lat"]),
                    "batch_index": int(batch_index),
                    "error": str(exc),
                }
                for _, point in batch_points.iterrows()
            ]
            return {
                "batch_failed": True,
                "completed_points": 0,
                "failed_points": failed_points,
                "missing_output_points": len(batch_points),
            }

        actual_file_map = {path.stem: path for path in batch_host_dir.glob("*.csv")}
        requested_ids = set(batch_points["point_id"].astype(str))
        completed_points = 0
        missing_output_points = 0
        failed_points: list[dict[str, object]] = []

        points_dir.mkdir(parents=True, exist_ok=True)
        for _, point in batch_points.iterrows():
            point_id = str(point["point_id"])
            source_file = actual_file_map.get(point_id)
            if source_file is None:
                missing_output_points += 1
                failed_points.append(
                    {
                        "point_index": int(point["point_index"]),
                        "point_id": point_id,
                        "lon": float(point["lon"]),
                        "lat": float(point["lat"]),
                        "batch_index": int(batch_index),
                        "error": "Missing point output file in batch result",
                    }
                )
                continue

            try:
                normalized = normalize_point_output(
                    source_file,
                    point_id=point_id,
                    lon=float(point["lon"]),
                    lat=float(point["lat"]),
                )
                normalized.to_csv(points_dir / f"{point_id}.csv", index=False)
                completed_points += 1
            except Exception as exc:
                failed_points.append(
                    {
                        "point_index": int(point["point_index"]),
                        "point_id": point_id,
                        "lon": float(point["lon"]),
                        "lat": float(point["lat"]),
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
        }
    finally:
        if batch_host_dir.exists():
            shutil.rmtree(batch_host_dir, ignore_errors=True)


def run_retry_round(
    retry_points: pd.DataFrame,
    *,
    batch_size: int,
    max_workers: int,
    batch_index_start: int,
    batch_name_prefix: str,
    base_url: str,
    batches_dir: Path,
    points_dir: Path,
    start_date: str,
    end_date: str,
    request_timeout_seconds: int,
) -> dict[str, object]:
    """Retry one set of failed points with a fixed batch size and worker count."""
    if retry_points.empty:
        return {
            "completed_points": 0,
            "failed_points": [],
            "missing_output_points": 0,
            "next_batch_index": batch_index_start,
        }

    completed_points = 0
    missing_output_points = 0
    failed_points: list[dict[str, object]] = []
    batches = iter_batches(retry_points, batch_size)

    for batch_offset, batch_points in enumerate(batches):
        batch_index = batch_index_start + batch_offset
        batch_host_dir = batches_dir / f"{batch_name_prefix}_{batch_index:06d}"
        batch_container_dir = host_to_container_path(batch_host_dir)
        result = process_batch(
            batch_points,
            batch_index=batch_index,
            base_url=base_url,
            batch_host_dir=batch_host_dir,
            batch_container_dir=batch_container_dir,
            points_dir=points_dir,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            request_timeout_seconds=request_timeout_seconds,
        )
        completed_points += int(result["completed_points"])
        missing_output_points += int(result["missing_output_points"])
        failed_points.extend(result["failed_points"])

    return {
        "completed_points": completed_points,
        "failed_points": dedupe_failures(failed_points),
        "missing_output_points": missing_output_points,
        "next_batch_index": batch_index_start + len(batches),
    }


def repair_failures(
    failures: list[dict[str, object]],
    *,
    base_url: str,
    batches_dir: Path,
    points_dir: Path,
    start_date: str,
    end_date: str,
    default_max_workers: int,
    request_timeout_seconds: int,
    batch_index_start: int,
) -> dict[str, object]:
    """Repair retryable failures after the shard's main batches finish."""
    deduped = dedupe_failures(failures)
    missing_failures = [
        row for row in deduped if classify_failure(str(row["error"])) == FAILURE_KIND_MISSING_OUTPUT
    ]
    timeout_failures = [
        row for row in deduped if classify_failure(str(row["error"])) == FAILURE_KIND_TIMEOUT
    ]
    non_retryable_failures = [
        row for row in deduped if classify_failure(str(row["error"])) == FAILURE_KIND_NON_RETRYABLE
    ]

    repair_attempted_points = len(missing_failures) + len(timeout_failures)
    repair_completed_points = 0
    repair_rounds_completed = 0
    missing_output_points = 0
    next_batch_index = batch_index_start
    final_failures: list[dict[str, object]] = list(non_retryable_failures)

    def _run_two_stage_repair(
        initial_failures: list[dict[str, object]],
        *,
        first_round_batch_size: int,
        first_round_max_workers: int,
        second_round_max_workers: int,
        prefix: str,
    ) -> list[dict[str, object]]:
        nonlocal repair_completed_points, repair_rounds_completed, missing_output_points, next_batch_index

        if not initial_failures:
            return []

        round_one_input = failure_rows_to_points_frame(initial_failures)
        repair_rounds_completed += 1
        round_one = run_retry_round(
            round_one_input,
            batch_size=first_round_batch_size,
            max_workers=first_round_max_workers,
            batch_index_start=next_batch_index,
            batch_name_prefix=f"{prefix}_r1",
            base_url=base_url,
            batches_dir=batches_dir,
            points_dir=points_dir,
            start_date=start_date,
            end_date=end_date,
            request_timeout_seconds=request_timeout_seconds,
        )
        next_batch_index = int(round_one["next_batch_index"])
        missing_output_points += int(round_one["missing_output_points"])
        round_one_remaining = dedupe_failures(round_one["failed_points"])
        repair_completed_points += len(initial_failures) - len(round_one_remaining)

        if not round_one_remaining:
            return []

        repair_rounds_completed += 1
        round_two = run_retry_round(
            failure_rows_to_points_frame(round_one_remaining),
            batch_size=1,
            max_workers=second_round_max_workers,
            batch_index_start=next_batch_index,
            batch_name_prefix=f"{prefix}_r2",
            base_url=base_url,
            batches_dir=batches_dir,
            points_dir=points_dir,
            start_date=start_date,
            end_date=end_date,
            request_timeout_seconds=request_timeout_seconds,
        )
        next_batch_index = int(round_two["next_batch_index"])
        missing_output_points += int(round_two["missing_output_points"])
        round_two_remaining = dedupe_failures(round_two["failed_points"])
        repair_completed_points += len(round_one_remaining) - len(round_two_remaining)
        return round_two_remaining

    final_failures.extend(
        _run_two_stage_repair(
            missing_failures,
            first_round_batch_size=3,
            first_round_max_workers=default_max_workers,
            second_round_max_workers=default_max_workers,
            prefix="repair_missing",
        )
    )
    final_failures.extend(
        _run_two_stage_repair(
            timeout_failures,
            first_round_batch_size=3,
            first_round_max_workers=1,
            second_round_max_workers=1,
            prefix="repair_timeout",
        )
    )

    final_failures = dedupe_failures(final_failures)
    return {
        "final_failures": final_failures,
        "repair_attempted_points": repair_attempted_points,
        "repair_completed_points": repair_completed_points,
        "repair_failed_points": len(final_failures) - len(non_retryable_failures),
        "repair_rounds_completed": repair_rounds_completed,
        "missing_output_points": missing_output_points,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download historical daily GEE/ERA5 weather for regional grid points."
    )
    parser.add_argument(
        "--points-path",
        type=Path,
        default=DEFAULT_POINTS_PATH,
        help=f"Input point table path (default: {DEFAULT_POINTS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory under data/processed (default: {DEFAULT_OUTPUT_DIR})",
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
        "--num-shards",
        type=int,
        default=DEFAULT_NUM_SHARDS,
        help=f"Total number of shards/devices (default: {DEFAULT_NUM_SHARDS})",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this device.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing final point files.")
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap on the number of points processed within this shard.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Number of points per GEE batch request (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"max_workers passed to /gee/meteo (default: {DEFAULT_MAX_WORKERS}).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Local meteo API base URL (default: {DEFAULT_BASE_URL}).")
    parser.add_argument("--request-timeout-seconds", type=int, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help=f"Per-batch request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS}).")
    return parser


def write_summary(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON summary payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_batches(points: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Split a point frame into stable fixed-size batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [points.iloc[start : start + batch_size].reset_index(drop=True) for start in range(0, len(points), batch_size)]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    points = prepare_points(args.points_path)
    shard_points = select_points_for_shard(
        points,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        max_points=args.max_points,
    )

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / f"shard_{int(args.shard_index):02d}"
    points_dir = shard_dir / "points"
    batches_dir = shard_dir / "batches"
    manifest_path = shard_dir / "manifest.csv"
    summary_path = shard_dir / "download_summary.json"
    failures_path = shard_dir / "failed_points.csv"

    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_points.to_csv(manifest_path, index=False)

    todo_rows = []
    skipped_complete_points = 0
    for _, point in shard_points.iterrows():
        point_path = points_dir / f"{point['point_id']}.csv"
        if not bool(args.overwrite) and point_file_is_complete(point_path):
            skipped_complete_points += 1
            continue
        todo_rows.append(point.to_dict())

    todo_points = pd.DataFrame(todo_rows, columns=shard_points.columns)
    failures: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "points_path": str(args.points_path),
        "output_dir": str(shard_dir),
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "batch_size": int(args.batch_size),
        "max_workers": int(args.max_workers),
        "total_points_in_shard": int(len(shard_points)),
        "completed_points": 0,
        "skipped_complete_points": int(skipped_complete_points),
        "failed_points": 0,
        "completed_batches": 0,
        "failed_batches": 0,
        "missing_output_points": 0,
        "repair_attempted_points": 0,
        "repair_completed_points": 0,
        "repair_failed_points": 0,
        "repair_rounds_completed": 0,
        "manifest_path": str(manifest_path),
        "failures_path": str(failures_path),
        "last_processed_point_id": None,
    }
    write_summary(summary_path, summary)
    pd.DataFrame(columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

    if todo_points.empty:
        if batches_dir.exists():
            try:
                batches_dir.rmdir()
            except OSError:
                pass
        print(f"Manifest: {manifest_path}")
        print(f"Summary: {summary_path}")
        print(f"Failures: {failures_path}")
        return 0

    batches = iter_batches(todo_points, int(args.batch_size))
    for batch_idx, batch_points in enumerate(batches, start=1):
        batch_host_dir = batches_dir / f"batch_{batch_idx:06d}"
        batch_container_dir = host_to_container_path(batch_host_dir)
        result = process_batch(
            batch_points,
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

        if bool(result["batch_failed"]):
            summary["failed_batches"] = int(summary["failed_batches"]) + 1
        else:
            summary["completed_batches"] = int(summary["completed_batches"]) + 1

        batch_failures = result["failed_points"]
        failures.extend(batch_failures)
        summary["failed_points"] = len(failures)
        if not batch_points.empty:
            summary["last_processed_point_id"] = str(batch_points["point_id"].iloc[-1])

        write_summary(summary_path, summary)
        pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

        print(
            f"[batch {batch_idx}/{len(batches)}] "
            f"completed={result['completed_points']} failed={len(batch_failures)} "
            f"missing={result['missing_output_points']}"
        )

    repair = repair_failures(
        failures,
        base_url=str(args.base_url),
        batches_dir=batches_dir,
        points_dir=points_dir,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        default_max_workers=int(args.max_workers),
        request_timeout_seconds=int(args.request_timeout_seconds),
        batch_index_start=len(batches) + 1,
    )
    final_failures = dedupe_failures(repair["final_failures"])
    summary["completed_points"] = int(summary["completed_points"]) + int(repair["repair_completed_points"])
    summary["missing_output_points"] = int(summary["missing_output_points"]) + int(repair["missing_output_points"])
    summary["failed_points"] = len(final_failures)
    summary["repair_attempted_points"] = int(repair["repair_attempted_points"])
    summary["repair_completed_points"] = int(repair["repair_completed_points"])
    summary["repair_failed_points"] = int(repair["repair_failed_points"])
    summary["repair_rounds_completed"] = int(repair["repair_rounds_completed"])
    write_summary(summary_path, summary)
    pd.DataFrame(final_failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

    if batches_dir.exists():
        try:
            batches_dir.rmdir()
        except OSError:
            pass

    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Failures: {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
