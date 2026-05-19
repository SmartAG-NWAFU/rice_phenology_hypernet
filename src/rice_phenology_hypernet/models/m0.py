from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rice_phenology_hypernet.data.daylength import DayLengthCalculator
from rice_phenology_hypernet.models.physics import (
    oryza2000_photo_response,
    trapezoidal_temperature_response,
)


STAGE_NAMES = ["tillering", "jointing", "booting", "heading", "maturity"]
THRESHOLD_COLUMNS = [
    "th_reviving_tillering",
    "th_tillering_jointing",
    "th_jointing_booting",
    "th_booting_heading",
    "th_heading_maturity",
]


@dataclass
class M0Parameters:
    t_base: float = 8.0
    t_opt_low: float = 25.0
    t_opt_high: float = 35.0
    t_cei: float = 42.0
    p_sens: float = 0.2
    p_crit: float = 12.5


class M0PhenologyModel:
    def __init__(self, params: M0Parameters | None = None):
        self.params = params or M0Parameters()
        self.thresholds = {name: np.nan for name in THRESHOLD_COLUMNS}
        self.daylength = DayLengthCalculator()

    def _build_weather_index(self, weather_df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
        return {
            (int(sid), int(year)): group.sort_values("Date").reset_index(drop=True)
            for (sid, year), group in weather_df.groupby(["SID", "year"])
        }

    def _prepare_weather(self, weather_df: pd.DataFrame, latitude: float) -> pd.DataFrame:
        df = weather_df.copy()
        df["thermal"] = trapezoidal_temperature_response(
            df["TemAver"].to_numpy(dtype=float),
            t_base=self.params.t_base,
            t_opt_low=self.params.t_opt_low,
            t_opt_high=self.params.t_opt_high,
            t_cei=self.params.t_cei,
        )
        df["daylength"] = [
            self.daylength.day_length(d.year, d.month, d.day, latitude) for d in df["Date"]
        ]
        df["photo"] = [
            oryza2000_photo_response(dl, self.params.p_crit, self.params.p_sens)
            for dl in df["daylength"]
        ]
        return df

    def _simulate_stage_doys(self, weather_df: pd.DataFrame, latitude: float, reviving_doy: float, thresholds: list[float]) -> list[float]:
        df = self._prepare_weather(weather_df, latitude)
        df["doy"] = df["Date"].dt.dayofyear
        df = df[df["doy"] >= reviving_doy].copy()

        predictions = []
        acc = 0.0
        current_stage = 0
        for _, row in df.iterrows():
            factor = 1.0 if current_stage < 2 or current_stage > 3 else row["photo"]
            acc += row["thermal"] * factor
            if acc >= sum(thresholds[: current_stage + 1]):
                predictions.append(float(row["doy"]))
                current_stage += 1
                if current_stage == len(thresholds):
                    break
        while len(predictions) < len(thresholds):
            predictions.append(np.nan)
        return predictions

    def collect_threshold_samples(self, weather_df: pd.DataFrame, phenology_df: pd.DataFrame) -> pd.DataFrame:
        weather_index = self._build_weather_index(weather_df)
        rows = []
        for _, row in phenology_df.iterrows():
            key = (int(row["SID"]), int(row["year"]))
            if key not in weather_index:
                continue
            weather = self._prepare_weather(weather_index[key], float(row["lat"]))
            reviving = pd.to_datetime(row["reviving date"], errors="coerce")
            if pd.isna(reviving):
                continue
            weather = weather[weather["Date"] >= reviving].copy()
            if weather.empty:
                continue
            weather["factor"] = 1.0
            jointing = pd.to_datetime(row.get("jointing date"), errors="coerce")
            heading = pd.to_datetime(row.get("heading date"), errors="coerce")
            if pd.notna(jointing) and pd.notna(heading):
                mask = (weather["Date"] > jointing) & (weather["Date"] <= heading)
                weather.loc[mask, "factor"] = weather.loc[mask, "photo"]
            weather["daily_dev"] = weather["thermal"] * weather["factor"]
            weather["cum_dev"] = weather["daily_dev"].cumsum()
            cum_map = weather.set_index("Date")["cum_dev"]
            stage_dates = [pd.to_datetime(row.get(f"{stage} date"), errors="coerce") for stage in STAGE_NAMES]
            cumulative = [cum_map.get(stage_date, np.nan) if pd.notna(stage_date) else np.nan for stage_date in stage_dates]

            diffs = []
            prev = 0.0
            valid = True
            for value in cumulative:
                if pd.isna(value):
                    valid = False
                    diffs.append(np.nan)
                    continue
                diff = float(value - prev)
                if diff <= 0:
                    valid = False
                    diffs.append(np.nan)
                else:
                    diffs.append(diff)
                    prev = float(value)
            if not valid:
                continue
            rows.append(
                {
                    "SID": int(row["SID"]),
                    "year": int(row["year"]),
                    "latitude": float(row["lat"]),
                    "longitude": float(row["lon"]),
                    "altitude": float(row["elevation"]),
                    "transplanting_date": pd.to_datetime(row.get("transplanting date"), errors="coerce"),
                    "reviving_date": reviving,
                    "source": "m0_inversion",
                    **dict(zip(THRESHOLD_COLUMNS, diffs)),
                }
            )
        return pd.DataFrame(rows)

    def fit(self, weather_df: pd.DataFrame, phenology_df: pd.DataFrame) -> dict[str, float]:
        threshold_df = self.collect_threshold_samples(weather_df, phenology_df)
        for column in THRESHOLD_COLUMNS:
            self.thresholds[column] = round(float(threshold_df[column].round(2).median()), 2)
        return dict(self.thresholds)

    def predict_one(self, weather_df: pd.DataFrame, sample: pd.Series) -> list[float]:
        thresholds = [self.thresholds[column] for column in THRESHOLD_COLUMNS]
        return self._simulate_stage_doys(
            weather_df=weather_df,
            latitude=float(sample["latitude"] if "latitude" in sample else sample["lat"]),
            reviving_doy=float(sample["obs_reviving"] if "obs_reviving" in sample else pd.to_datetime(sample["reviving date"]).dayofyear),
            thresholds=thresholds,
        )


class M0TPhenologyModel:
    """温度-only 基线模型（不考虑光周期效应）。

    与 M0PhenologyModel 的关键差异：
    - 只使用 trapezoidal_temperature_response 计算日尺度发育贡献
    - 所有阶段统一使用 factor = 1.0，不对 booting/heading 施加光周期缩放
    - 阈值反演使用温度-only 的历史累积
    """

    def __init__(self, params: M0Parameters | None = None):
        self.params = params or M0Parameters()
        self.thresholds = {name: np.nan for name in THRESHOLD_COLUMNS}

    def _build_weather_index(self, weather_df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
        return {
            (int(sid), int(year)): group.sort_values("Date").reset_index(drop=True)
            for (sid, year), group in weather_df.groupby(["SID", "year"])
        }

    def _prepare_weather_t(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """只计算 thermal，不计算 photo。"""
        df = weather_df.copy()
        df["thermal"] = trapezoidal_temperature_response(
            df["TemAver"].to_numpy(dtype=float),
            t_base=self.params.t_base,
            t_opt_low=self.params.t_opt_low,
            t_opt_high=self.params.t_opt_high,
            t_cei=self.params.t_cei,
        )
        return df

    def _simulate_stage_doys_t(self, weather_df: pd.DataFrame, reviving_doy: float, thresholds: list[float]) -> list[float]:
        """温度-only 模拟：所有阶段 factor = 1.0。"""
        df = self._prepare_weather_t(weather_df)
        df["doy"] = df["Date"].dt.dayofyear
        df = df[df["doy"] >= reviving_doy].copy()

        predictions = []
        acc = 0.0
        current_stage = 0
        for _, row in df.iterrows():
            # 温度-only：所有阶段统一使用 factor = 1.0
            acc += row["thermal"]
            if acc >= sum(thresholds[: current_stage + 1]):
                predictions.append(float(row["doy"]))
                current_stage += 1
                if current_stage == len(thresholds):
                    break
        while len(predictions) < len(thresholds):
            predictions.append(np.nan)
        return predictions

    def collect_threshold_samples_t(self, weather_df: pd.DataFrame, phenology_df: pd.DataFrame) -> pd.DataFrame:
        """温度-only 阈值反演：历史累积只使用 thermal。"""
        weather_index = self._build_weather_index(weather_df)
        rows = []
        for _, row in phenology_df.iterrows():
            key = (int(row["SID"]), int(row["year"]))
            if key not in weather_index:
                continue
            weather = self._prepare_weather_t(weather_index[key])
            reviving = pd.to_datetime(row["reviving date"], errors="coerce")
            if pd.isna(reviving):
                continue
            weather = weather[weather["Date"] >= reviving].copy()
            if weather.empty:
                continue
            # 温度-only：daily_dev = thermal（所有阶段统一）
            weather["daily_dev"] = weather["thermal"]
            weather["cum_dev"] = weather["daily_dev"].cumsum()
            cum_map = weather.set_index("Date")["cum_dev"]
            stage_dates = [pd.to_datetime(row.get(f"{stage} date"), errors="coerce") for stage in STAGE_NAMES]
            cumulative = [cum_map.get(stage_date, np.nan) if pd.notna(stage_date) else np.nan for stage_date in stage_dates]

            diffs = []
            prev = 0.0
            valid = True
            for value in cumulative:
                if pd.isna(value):
                    valid = False
                    diffs.append(np.nan)
                    continue
                diff = float(value - prev)
                if diff <= 0:
                    valid = False
                    diffs.append(np.nan)
                else:
                    diffs.append(diff)
                    prev = float(value)
            if not valid:
                continue
            rows.append(
                {
                    "SID": int(row["SID"]),
                    "year": int(row["year"]),
                    "latitude": float(row["lat"]),
                    "longitude": float(row["lon"]),
                    "altitude": float(row["elevation"]),
                    "transplanting_date": pd.to_datetime(row.get("transplanting date"), errors="coerce"),
                    "reviving_date": reviving,
                    "source": "m0_t_inversion",
                    **dict(zip(THRESHOLD_COLUMNS, diffs)),
                }
            )
        return pd.DataFrame(rows)

    def fit(self, weather_df: pd.DataFrame, phenology_df: pd.DataFrame) -> dict[str, float]:
        """温度-only 阈值拟合。"""
        threshold_df = self.collect_threshold_samples_t(weather_df, phenology_df)
        for column in THRESHOLD_COLUMNS:
            self.thresholds[column] = round(float(threshold_df[column].round(2).median()), 2)
        return dict(self.thresholds)

    def predict_one(self, weather_df: pd.DataFrame, sample: pd.Series) -> list[float]:
        thresholds = [self.thresholds[column] for column in THRESHOLD_COLUMNS]
        return self._simulate_stage_doys_t(
            weather_df=weather_df,
            reviving_doy=float(sample["obs_reviving"] if "obs_reviving" in sample else pd.to_datetime(sample["reviving date"]).dayofyear),
            thresholds=thresholds,
        )
