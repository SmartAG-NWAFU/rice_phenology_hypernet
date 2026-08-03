from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RawDataPaths:
    """Explicit source paths for private station data."""

    weather: Path
    phenology: Path


@dataclass(frozen=True)
class PreparedDataPaths:
    weather: Path
    phenology: Path
    modeling_dataset: Path
    threshold_samples: Path


@dataclass(frozen=True)
class ExperimentBundle:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    parameters: pd.DataFrame
