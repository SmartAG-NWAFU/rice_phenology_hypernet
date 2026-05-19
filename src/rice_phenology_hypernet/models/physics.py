from __future__ import annotations

import numpy as np


def trapezoidal_temperature_response(
    temperature,
    t_base: float = 8.0,
    t_opt_low: float = 25.0,
    t_opt_high: float = 35.0,
    t_cei: float = 42.0,
):
    max_rate = t_opt_low - t_base
    return np.interp(
        temperature,
        [t_base, t_opt_low, t_opt_high, t_cei],
        [0.0, max_rate, max_rate, 0.0],
    )


def oryza2000_photo_response(day_length: float, p_crit: float = 12.5, p_sens: float = 0.2) -> float:
    if day_length < p_crit:
        return 1.0
    return float(np.clip(1.0 - (day_length - p_crit) * p_sens, 0.0, 1.0))
