"""Shared scientific contracts for daily development-rate experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DVR_STAGE_NAMES = ("tillering", "jointing", "booting", "heading", "maturity")
PHOTO_SENSITIVE_STAGES = frozenset({"booting", "heading"})
DEFAULT_WEATHER_FEATURES = (
    "TemAver",
    "TemMin",
    "TemMax",
    "daylength",
    "Precipitation",
)
PAPER_MODEL_NAMES = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")


@dataclass(frozen=True)
class StageInputs:
    """Daily inputs for one phenological-stage rollout."""

    doy: np.ndarray
    mask: np.ndarray
    model_inputs: object | None


@dataclass(frozen=True)
class StageRolloutResult:
    """Daily correction and threshold-crossing result for one stage."""

    completion_doy: float
    next_start_doy: float
    corrected_dvr: np.ndarray
    cumulative_progress: np.ndarray


__all__ = [
    "DEFAULT_WEATHER_FEATURES",
    "DVR_STAGE_NAMES",
    "PAPER_MODEL_NAMES",
    "PHOTO_SENSITIVE_STAGES",
    "StageInputs",
    "StageRolloutResult",
]
