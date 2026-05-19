#!/usr/bin/env python3
"""Visualize Middle rice phenology dates GeoTIFF files."""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path

# File paths
DATA_DIR = Path(__file__).parent.parent.parent / "data/external/china_rice_calendar/dataverse_v8/rice_pixels/2003_2022"

FILES = {
    "transplanting": "Middle_rice_transplanting_dates_2003_2022_rice_pixels.tif",
    "heading": "Middle_rice_heading_dates_2003_2022_rice_pixels.tif",
    "maturity": "Middle_rice_maturity_dates_2003_2022_rice_pixels.tif",
}

LABELS = {
    "transplanting": "Transplanting",
    "heading": "Heading",
    "maturity": "Maturity",
}


def read_geotiff(path: Path) -> tuple[np.ndarray, dict]:
    """Read GeoTIFF and return data + metadata."""
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "nodata": src.nodata,
        }
    return data, meta


def main():
    # Read all three files
    data_dict = {}
    meta = None
    
    for stage, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        data, meta = read_geotiff(path)
        data_dict[stage] = data
        print(f"{stage}: shape={data.shape}, nodata={meta['nodata']}, "
              f"valid pixels={np.sum(data != meta['nodata'])}")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    nodata = meta["nodata"]
    
    for ax, (stage, data) in zip(axes, data_dict.items()):
        # Mask nodata values
        masked = np.ma.masked_where(data == nodata, data)
        
        # Plot
        im = ax.imshow(masked, cmap="YlOrRd", interpolation="none")
        ax.set_title(f"Middle Rice - {LABELS[stage]}", fontsize=12)
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Day of Year", fontsize=10)
        
        # Stats annotation
        valid_data = data[data != nodata]
        if len(valid_data) > 0:
            stats_text = f"min={valid_data.min():.0f}\nmax={valid_data.max():.0f}\nmean={valid_data.mean():.0f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.suptitle("China Middle Rice Phenology Calendar (2003-2022)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = DATA_DIR / "middle_rice_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())