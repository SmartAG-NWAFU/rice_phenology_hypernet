from __future__ import annotations

import datetime as dt
import math

import numpy as np


class DayLengthCalculator:
    def day_length(self, year: int, month: int, day: int, latitude: float) -> float:
        doy = dt.date(year, month, day).timetuple().tm_yday
        phi = np.deg2rad(latitude)
        delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
        p = np.deg2rad(-0.833)
        cos_w = (np.sin(p) - np.sin(phi) * np.sin(delta)) / (
            np.cos(phi) * np.cos(delta)
        )
        if cos_w > 1:
            return 0.0
        if cos_w < -1:
            return 24.0
        w = math.acos(cos_w)
        return 24.0 * w / math.pi
