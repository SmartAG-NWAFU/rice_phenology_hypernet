#!/usr/bin/env python3
"""
Download daily temperature data from Open-Meteo API.
Supports incremental download with smart date range detection.
"""

import argparse
import datetime as dt
import os
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import requests

# ============================================================================
# Path Configuration
# ============================================================================

DATA_DIR = Path("/Users/binchen/workshop/rice_phenology_hypernet/data")

# ============================================================================
# Weather Class (Embedded from open_meteo.py)
# ============================================================================


class Weather:
    """
    Weather data retrieval class using the official Open-Meteo API.

    Notes
    -----
    - Historical daily/hourly data are fetched from the Open-Meteo
      Historical Weather API.
    - Returned data are converted to pandas.DataFrame.
    - The user provides latitude, longitude, date range, and weather variables.
    """

    BASE_URL_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
    MAX_RETRIES = int(os.getenv("OPEN_METEO_MAX_RETRIES", "8"))
    BACKOFF_BASE_SECONDS = float(os.getenv("OPEN_METEO_BACKOFF_BASE_SECONDS", "2.0"))
    MAX_BACKOFF_SECONDS = float(os.getenv("OPEN_METEO_MAX_BACKOFF_SECONDS", "120.0"))
    MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS", "1.2"))
    _LAST_REQUEST_TS: float | None = None

    COMMON_DAILY_VARS = [
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "sunrise",
        "sunset",
        "daylight_duration",
        "sunshine_duration",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]

    @staticmethod
    def _validate_date(date_str: str) -> None:
        """Validate date string in YYYY-MM-DD format."""
        try:
            dt.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.") from e

    @staticmethod
    def _validate_coordinates(latitude: float, longitude: float) -> None:
        """Validate geographic coordinates."""
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Latitude must be within [-90, 90], got {latitude}.")
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Longitude must be within [-180, 180], got {longitude}.")

    @staticmethod
    def _normalize_vars(variables: Sequence[str]) -> str:
        """Convert a sequence of variable names to a comma-separated string."""
        if not variables:
            raise ValueError("At least one weather variable must be provided.")
        cleaned = [str(v).strip() for v in variables if str(v).strip()]
        if not cleaned:
            raise ValueError("No valid weather variables were provided.")
        return ",".join(cleaned)

    @classmethod
    def _respect_rate_limit(cls) -> None:
        if cls.MIN_REQUEST_INTERVAL_SECONDS <= 0:
            return

        now = time.monotonic()
        if cls._LAST_REQUEST_TS is not None:
            elapsed = now - cls._LAST_REQUEST_TS
            sleep_seconds = cls.MIN_REQUEST_INTERVAL_SECONDS - elapsed
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        cls._LAST_REQUEST_TS = time.monotonic()

    @classmethod
    def _retry_delay_seconds(
        cls,
        attempt: int,
        response: requests.Response | None = None,
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                retry_after = retry_after.strip()
                if retry_after.isdigit():
                    return min(float(retry_after), cls.MAX_BACKOFF_SECONDS)
                try:
                    retry_after_dt = parsedate_to_datetime(retry_after)
                    if retry_after_dt.tzinfo is None:
                        retry_after_dt = retry_after_dt.replace(tzinfo=dt.timezone.utc)
                    now = dt.datetime.now(tz=dt.timezone.utc)
                    retry_after_seconds = (retry_after_dt - now).total_seconds()
                    if retry_after_seconds > 0:
                        return min(retry_after_seconds, cls.MAX_BACKOFF_SECONDS)
                except (TypeError, ValueError, OverflowError):
                    pass

        exp_backoff = cls.BACKOFF_BASE_SECONDS * (2 ** attempt)
        jitter = random.uniform(0, cls.BACKOFF_BASE_SECONDS / 2)
        return min(exp_backoff + jitter, cls.MAX_BACKOFF_SECONDS)

    @classmethod
    def _request(cls, params: dict, timeout: int = 60) -> dict:
        """Send a request to Open-Meteo with retry and backoff."""
        last_error: Exception | None = None

        for attempt in range(cls.MAX_RETRIES + 1):
            cls._respect_rate_limit()
            try:
                response = requests.get(cls.BASE_URL_HISTORICAL, params=params, timeout=timeout)
            except requests.RequestException as e:
                last_error = e
                if attempt >= cls.MAX_RETRIES:
                    break
                wait_seconds = cls._retry_delay_seconds(attempt)
                print(
                    "Open-Meteo request error, retrying "
                    f"({attempt + 1}/{cls.MAX_RETRIES}) in {wait_seconds:.1f}s: {e}"
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                try:
                    response.raise_for_status()
                except requests.RequestException as e:
                    last_error = e
                if attempt >= cls.MAX_RETRIES:
                    break
                wait_seconds = cls._retry_delay_seconds(attempt, response=response)
                print(
                    "Open-Meteo rate limit/server error, retrying "
                    f"({attempt + 1}/{cls.MAX_RETRIES}) in {wait_seconds:.1f}s: "
                    f"HTTP {response.status_code}"
                )
                time.sleep(wait_seconds)
                continue

            try:
                response.raise_for_status()
                payload = response.json()
                return payload
            except requests.RequestException as e:
                raise RuntimeError(
                    f"Failed to fetch data from Open-Meteo. Params={params}. Error={e}"
                ) from e

        raise RuntimeError(
            "Failed to fetch data from Open-Meteo after retries. "
            f"Params={params}. Retries={cls.MAX_RETRIES}. LastError={last_error}"
        ) from last_error

    @staticmethod
    def _to_dataframe(payload: dict, section: str) -> pd.DataFrame:
        """
        Convert Open-Meteo 'hourly' or 'daily' JSON section to DataFrame.
        """
        if section not in payload:
            raise ValueError(
                f"Response does not contain '{section}'. Response keys: {list(payload.keys())}"
            )

        block = payload[section]
        if "time" not in block:
            raise ValueError(f"Response '{section}' block does not contain 'time'.")

        df = pd.DataFrame(block)
        time_col = "time"

        if section == "hourly":
            df.rename(columns={time_col: "DateTime"}, inplace=True)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
        else:
            df.rename(columns={time_col: "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.date

        return df

    @classmethod
    def get_daily(
        cls,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Sequence[str],
        timezone: str = "Asia/Shanghai",
        temperature_unit: str = "celsius",
        wind_speed_unit: str = "ms",
        precipitation_unit: str = "mm",
    ) -> pd.DataFrame:
        """
        Get daily weather data for a location and date range.
        """
        cls._validate_coordinates(latitude, longitude)
        cls._validate_date(start_date)
        cls._validate_date(end_date)

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": cls._normalize_vars(variables),
            "timezone": timezone,
            "temperature_unit": temperature_unit,
            "wind_speed_unit": wind_speed_unit,
            "precipitation_unit": precipitation_unit,
        }

        payload = cls._request(params)
        return cls._to_dataframe(payload, section="daily")

    @classmethod
    def get_daily_agri_weather(cls, latitude, longitude, start_date, end_date):
        """Get daily agricultural weather data."""
        return cls.get_daily(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=[
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "shortwave_radiation_sum",
            ],
        )


# ============================================================================
# Download Configuration
# ============================================================================

TEMP_COLUMNS = ["TemAver", "TemMin", "TemMax", "Precipitation", "Radiation"]
OUTPUT_COLUMNS = ["SID", "year", "Date", *TEMP_COLUMNS]


# ============================================================================
# Helper Functions
# ============================================================================


def _load_stations(catalog_path: Path) -> pd.DataFrame:
    """Load station catalog from Excel file."""
    df = pd.read_excel(catalog_path)
    if "station ID" in df.columns:
        df = df.rename(columns={"station ID": "SID"})

    required = {"SID", "lat", "lon"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in catalog: {missing}")

    df = df[list(required)].copy()
    df = df.dropna(subset=["SID", "lat", "lon"])
    df["SID"] = df["SID"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df.drop_duplicates(subset=["SID", "lat", "lon"]).reset_index(drop=True)

    duplicates = df[df.duplicated(subset=["SID"], keep=False)]
    if not duplicates.empty:
        mismatched = duplicates.groupby("SID")[["lat", "lon"]].nunique()
        mismatched = mismatched[
            (mismatched["lat"] > 1) | (mismatched["lon"] > 1)
        ]
        if not mismatched.empty:
            preview = ", ".join(mismatched.index.astype(str)[:5])
            suffix = "..." if len(mismatched) > 5 else ""
            print(
                "Warning: multiple lat/lon entries for some stations; "
                f"keeping the first occurrence (e.g. {preview}{suffix})."
            )
        df = df.sort_values("SID").drop_duplicates(subset=["SID"], keep="first").reset_index(drop=True)

    return df


def _normalize_temperature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize temperature data frame columns and types."""
    if df.empty:
        return df

    data = df.copy()
    data["SID"] = data["SID"].astype(str)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in TEMP_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["SID", "Date", *TEMP_COLUMNS])
    data["year"] = data["Date"].dt.year.astype(int)
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    return data[OUTPUT_COLUMNS]


def _load_existing(output_path: Path) -> pd.DataFrame | None:
    """Load existing data without date filtering."""
    if not output_path.exists():
        return None

    existing = pd.read_csv(output_path)
    if existing.empty:
        return None

    required = {"SID", "Date"}
    missing = sorted(required - set(existing.columns))
    if missing:
        raise ValueError(f"Missing required columns in existing data: {missing}")

    missing_temps = [col for col in TEMP_COLUMNS if col not in existing.columns]
    if missing_temps:
        print(
            "Warning: existing data missing temperature columns "
            f"{missing_temps}; they will be refetched."
        )

    existing = _normalize_temperature_frame(existing)
    if existing.empty:
        return None

    return existing


def _fetch_daily_temperature_range(
    sid: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily temperature data for a station and date range."""
    wd = Weather.get_daily_agri_weather(lat, lon, start_date, end_date)
    wd = wd.rename(
        columns={
            "temperature_2m_mean": "TemAver",
            "temperature_2m_min": "TemMin",
            "temperature_2m_max": "TemMax",
            "precipitation_sum": "Precipitation",
            "shortwave_radiation_sum": "Radiation",
        }
    )
    if "TemAver" not in wd.columns and {"TemMin", "TemMax"}.issubset(wd.columns):
        wd["TemAver"] = (
            pd.to_numeric(wd["TemMin"], errors="coerce")
            + pd.to_numeric(wd["TemMax"], errors="coerce")
        ) / 2
    wd["SID"] = sid
    wd["Date"] = pd.to_datetime(wd["Date"], errors="coerce")
    for col in TEMP_COLUMNS:
        if col not in wd.columns:
            wd[col] = pd.NA
        wd[col] = pd.to_numeric(wd[col], errors="coerce")
    wd = wd.dropna(subset=["Date", *TEMP_COLUMNS])
    wd["year"] = wd["Date"].dt.year.astype(int)
    wd["Date"] = wd["Date"].dt.strftime("%Y-%m-%d")
    return wd[OUTPUT_COLUMNS]


def _get_missing_date_ranges(
    df: pd.DataFrame,
    target_start: str,
    target_end: str,
) -> list[tuple[str, str]]:
    """
    Find missing date ranges within the target range.
    
    Returns list of (start_date, end_date) tuples that need to be fetched.
    """
    expected = pd.date_range(target_start, target_end, freq="D")
    
    if df.empty:
        return [(target_start, target_end)]
    
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    temp_cols = [col for col in TEMP_COLUMNS if col in data.columns]
    if temp_cols:
        data = data.dropna(subset=temp_cols)
    
    actual = pd.DatetimeIndex(data["Date"].dt.normalize().unique())
    missing = expected.difference(actual)
    
    if missing.empty:
        return []
    
    # Group consecutive missing dates into ranges
    dates = missing.sort_values()
    ranges = []
    range_start = dates[0]
    prev = dates[0]
    
    for current in dates[1:]:
        if (current - prev).days == 1:
            prev = current
            continue
        ranges.append((range_start, prev))
        range_start = current
        prev = current
    ranges.append((range_start, prev))
    
    return [
        (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        for start, end in ranges
    ]


def _merge_temperature_frames(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Merge two temperature frames, keeping latest values for duplicates."""
    if extra.empty:
        return base
    if base.empty:
        return extra
    combined = pd.concat([base, extra], ignore_index=True)
    return combined.drop_duplicates(subset=["SID", "year", "Date"], keep="last").reset_index(drop=True)


def _check_missing_daily_data(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Check for missing daily data per station-year."""
    if df.empty:
        return pd.DataFrame(
            columns=["SID", "year", "expected_days", "actual_days", "missing_days", "missing_dates"]
        )

    data = df.copy()
    data["SID"] = data["SID"].astype(str)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["SID", "Date", *TEMP_COLUMNS])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    data = data[(data["Date"] >= start) & (data["Date"] <= end)]
    data["year"] = data["Date"].dt.year.astype(int)

    years = range(start.year, end.year + 1)
    results = []
    for sid in sorted(data["SID"].unique()):
        sid_data = data[data["SID"] == sid]
        for year in years:
            expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            actual = pd.DatetimeIndex(
                sid_data[sid_data["year"] == year]["Date"].dt.normalize().unique()
            )
            missing = expected.difference(actual)
            if missing.empty:
                preview = ""
            else:
                preview = ",".join(missing.strftime("%Y-%m-%d")[:5])
                if len(missing) > 5:
                    preview = f"{preview}..."

            results.append(
                {
                    "SID": sid,
                    "year": year,
                    "expected_days": len(expected),
                    "actual_days": len(actual),
                    "missing_days": len(missing),
                    "missing_dates": preview,
                }
            )

    return pd.DataFrame(results)


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download daily temperature data from Open-Meteo API.",
        epilog=(
            "Examples:\n"
            "  # Download 1981-2010 (default)\n"
            "  python download_daily_temperature.py\n\n"
            "  # Extend to 2020 (incremental, skips existing 1981-2010)\n"
            "  python download_daily_temperature.py --end-date 2020-12-31\n\n"
            "  # Download only 2011-2020\n"
            "  python download_daily_temperature.py --start-date 2011-01-01 --end-date 2020-12-31\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        default=DATA_DIR / "raw" / "obser_pheno_catalog_origin.xlsx",
        help="Path to obser_pheno_catalog.xlsx",
    )
    parser.add_argument(
        "--output",
        default=DATA_DIR / "raw" / "daily_temperature.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--start-date",
        default="1981-01-01",
        help="Start date (YYYY-MM-DD). Default: 1981-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2010-12-31",
        help="End date (YYYY-MM-DD). Default: 2010-12-31",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output instead of resuming",
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=2.0,
        help="Minimum interval between Open-Meteo requests to avoid 429 limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum retry attempts for a failed Open-Meteo request.",
    )
    parser.add_argument(
        "--backoff-base-seconds",
        type=float,
        default=2.5,
        help="Base seconds for exponential backoff (with jitter).",
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=180.0,
        help="Maximum sleep seconds for each retry backoff.",
    )
    args = parser.parse_args()

    # Configure Weather class retry parameters
    Weather.MIN_REQUEST_INTERVAL_SECONDS = max(0.0, float(args.request_interval_seconds))
    Weather.MAX_RETRIES = max(0, int(args.max_retries))
    Weather.BACKOFF_BASE_SECONDS = max(0.1, float(args.backoff_base_seconds))
    Weather.MAX_BACKOFF_SECONDS = max(1.0, float(args.max_backoff_seconds))

    catalog_path = Path(args.catalog)
    output_path = Path(args.output)
    start_date = args.start_date
    end_date = args.end_date

    print(
        "Open-Meteo retry config: "
        f"interval={Weather.MIN_REQUEST_INTERVAL_SECONDS}s, "
        f"max_retries={Weather.MAX_RETRIES}, "
        f"backoff_base={Weather.BACKOFF_BASE_SECONDS}s, "
        f"max_backoff={Weather.MAX_BACKOFF_SECONDS}s"
    )
    print(f"Target date range: {start_date} to {end_date}")

    # Load stations
    stations = _load_stations(catalog_path)
    print(f"Loaded {len(stations)} stations from catalog")

    # Load existing data (without date filtering)
    existing_df = None
    existing_by_sid: dict[str, pd.DataFrame] = {}
    
    if output_path.exists() and not args.overwrite:
        existing_df = _load_existing(output_path)
        if existing_df is not None and not existing_df.empty:
            existing_by_sid = {
                sid: group.copy() for sid, group in existing_df.groupby("SID")
            }
            print(f"Loaded {len(existing_by_sid)} stations from existing data")

    # Process each station
    records = []
    total = len(stations)
    skipped = 0
    
    for idx, row in stations.iterrows():
        sid = str(row["SID"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        
        # Get existing data for this station
        existing_sid = existing_by_sid.get(sid)
        
        # Find missing date ranges within target range
        if existing_sid is not None and not existing_sid.empty:
            missing_ranges = _get_missing_date_ranges(existing_sid, start_date, end_date)
        else:
            missing_ranges = [(start_date, end_date)]
        
        if not missing_ranges:
            # Station is complete within target range
            skipped += 1
            continue
        
        # Fetch only missing date ranges
        fetched_frames = []
        for fetch_start, fetch_end in missing_ranges:
            try:
                print(f"  [{idx + 1}/{total}] {sid}: fetching {fetch_start} ~ {fetch_end}")
                fetched = _fetch_daily_temperature_range(sid, lat, lon, fetch_start, fetch_end)
                if not fetched.empty:
                    fetched_frames.append(fetched)
            except Exception as exc:
                print(f"  Failed {sid} ({lat}, {lon}) {fetch_start}~{fetch_end}: {exc}")
                continue
        
        # Merge fetched data with existing
        if fetched_frames:
            fetched_df = pd.concat(fetched_frames, ignore_index=True)
            if existing_sid is not None and not existing_sid.empty:
                merged = _merge_temperature_frames(existing_sid, fetched_df)
            else:
                merged = fetched_df
            records.append(merged)
            print(f"  [{idx + 1}/{total}] {sid}: +{len(fetched_df)} rows (total {len(merged)} for station)")
    
    print(f"Skipped {skipped} stations with complete data in target range")

    # Combine all data
    if records:
        new_df = pd.concat(records, ignore_index=True)
    else:
        new_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Merge with all existing data (not just target range)
    frames = []
    if existing_df is not None and not existing_df.empty:
        frames.append(existing_df)
    if not new_df.empty:
        frames.append(new_df)
    
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["SID", "year", "Date"], keep="last").reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined):,} rows to {output_path}")

    # Check for missing data in target range
    report = _check_missing_daily_data(combined, start_date, end_date)
    if report.empty:
        print("No data available for missing-date checks.")
        return

    missing_report = report[report["missing_days"] > 0]
    if missing_report.empty:
        print("All station-year records have complete daily coverage in target range.")
        return

    print(f"{len(missing_report)} station-year records have missing daily data:")
    for _, row in missing_report.iterrows():
        suffix = f" (e.g. {row['missing_dates']})" if row["missing_dates"] else ""
        print(
            f"- {row['SID']} {row['year']}: missing {row['missing_days']} days"
            f" (expected {row['expected_days']}, got {row['actual_days']}){suffix}"
        )


if __name__ == "__main__":
    main()