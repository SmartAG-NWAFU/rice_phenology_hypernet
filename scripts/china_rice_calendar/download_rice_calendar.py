#!/usr/bin/env python3
"""
Download ChinaRiceCalendar GeoTIFF files from Harvard Dataverse.

Downloads Early/Middle/Late × transplanting/heading/maturity ×
2003-2022 period rice_pixels files.
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Fixed configuration
DATAVERSE_API = "https://dataverse.harvard.edu/api"
DOI = "doi:10.7910/DVN/EUP8EY"
VERSION = 8

SEASONS = ("Early", "Middle", "Late")
STAGES = ("transplanting", "heading", "maturity")
PERIODS = ("2003_2007", "2008_2012", "2013_2017", "2018_2022", "2003_2022")
FILE_TYPE = "rice_pixels"

# Target filenames (45 files)
TARGET_FILES = [
    f"{season}_rice_{stage}_dates_{period}_{FILE_TYPE}.tif"
    for period in PERIODS
    for season in SEASONS
    for stage in STAGES
]
TARGET_FILE_ORDER = {filename: index for index, filename in enumerate(TARGET_FILES)}

# Default output directory (relative to project root)
DEFAULT_OUTPUT_DIR = "data/external/china_rice_calendar/dataverse_v8/rice_pixels"


def fetch_dataset_metadata() -> dict[str, Any]:
    """Fetch dataset metadata from Dataverse API."""
    url = f"{DATAVERSE_API}/datasets/:persistentId/?persistentId={DOI}"
    print(f"Fetching dataset metadata from: {url}")
    
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ChinaRiceCalendar-Downloader/1.0",
        },
    )
    with urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def find_target_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Find target files in dataset metadata."""
    data_files = metadata["data"]["latestVersion"]["files"]
    
    found = []
    for f in data_files:
        filename = f["dataFile"]["filename"]
        if filename in TARGET_FILE_ORDER:
            found.append({
                "filename": filename,
                "id": f["dataFile"]["id"],
                "size": f["dataFile"]["filesize"],
                "md5": f["dataFile"].get("md5", ""),
            })
    
    return sorted(found, key=lambda f: TARGET_FILE_ORDER[f["filename"]])


def extract_period(filename: str) -> str:
    """Return the period token embedded in a ChinaRiceCalendar filename."""
    for period in PERIODS:
        if f"_{period}_" in filename:
            return period
    raise ValueError(f"Could not infer period from filename: {filename}")


def local_output_path(output_dir: Path, filename: str) -> Path:
    """Place each period in its own subdirectory under output_dir."""
    return output_dir / extract_period(filename) / filename


def download_file(file_id: int, output_path: Path, expected_size: int) -> bool:
    """Download a file from Dataverse with atomic write."""
    url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    print(f"  Downloading from: {url}")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        req = Request(
            url,
            headers={"User-Agent": "ChinaRiceCalendar-Downloader/1.0"},
        )
        with urlopen(req, timeout=300) as response:
            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=output_path.parent)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(response.read())

                actual_size = os.path.getsize(temp_path)
                if actual_size != expected_size:
                    print(f"  ERROR: Size mismatch: {actual_size} vs {expected_size}")
                    os.unlink(temp_path)
                    return False
                
                # Atomic rename
                os.replace(temp_path, output_path)
                return True
            except Exception:
                # Cleanup temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        return False


def build_manifest(
    files: list[dict[str, Any]],
    output_dir: Path,
    download_time: str,
) -> dict[str, Any]:
    """Build download manifest."""
    return {
        "dataset_doi": DOI,
        "dataverse_version": VERSION,
        "seasons": list(SEASONS),
        "stages": list(STAGES),
        "periods": list(PERIODS),
        "file_type": FILE_TYPE,
        "download_time": download_time,
        "files": [
            {
                "filename": f["filename"],
                "dataverse_file_id": f["id"],
                "size_bytes": f["size"],
                "md5": f.get("md5", ""),
                "local_path": str(local_output_path(output_dir, f["filename"])),
            }
            for f in files
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download ChinaRiceCalendar GeoTIFF files from Harvard Dataverse"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without downloading",
    )
    args = parser.parse_args()

    # Determine output directory
    project_root = Path(__file__).parent.parent.parent
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = project_root / DEFAULT_OUTPUT_DIR

    # Fetch metadata
    metadata = fetch_dataset_metadata()
    
    # Find target files
    found_files = find_target_files(metadata)
    
    if len(found_files) != len(TARGET_FILES):
        found_names = {f["filename"] for f in found_files}
        missing = [f for f in TARGET_FILES if f not in found_names]
        print(f"WARNING: Found {len(found_files)}/{len(TARGET_FILES)} target files")
        print(f"Missing files: {missing}")
    
    print(f"\nFound {len(found_files)} target files:")
    for f in found_files:
        size_mb = f["size"] / (1024 * 1024)
        period = extract_period(f["filename"])
        print(f"  - {period}/{f['filename']} ({size_mb:.2f} MB, id={f['id']})")
    
    if args.dry_run:
        print("\n[DRY RUN] Would download the above files to:")
        print(f"  {output_dir}/<period>/")
        return 0
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
    download_time = datetime.now(timezone.utc).isoformat()
    downloaded = []
    
    for f in found_files:
        output_path = local_output_path(output_dir, f["filename"])
        
        # Check if file already exists with matching size
        if output_path.exists():
            existing_size = output_path.stat().st_size
            if existing_size == f["size"]:
                print(f"  Skipping (exists): {extract_period(f['filename'])}/{f['filename']}")
                downloaded.append(f)
                continue
        
        print(f"  Downloading: {extract_period(f['filename'])}/{f['filename']}")
        if download_file(f["id"], output_path, f["size"]):
            downloaded.append(f)
        else:
            print(f"  FAILED: {extract_period(f['filename'])}/{f['filename']}")
    
    # Write manifest
    manifest = build_manifest(found_files, output_dir, download_time)
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    
    print(f"\nManifest written to: {manifest_path}")
    print(f"Downloaded {len(downloaded)}/{len(found_files)} files")
    
    return 0 if len(downloaded) == len(found_files) else 1


if __name__ == "__main__":
    exit(main())
