from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rice_phenology_hypernet.data.daylength import DayLengthCalculator
from rice_phenology_hypernet.experiments.dvr_core import (
    MAX_TRANSITION_DAYS,
    PHOTO_SENSITIVE_STAGES,
)
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
TRANSITION_START_COLUMNS = [
    "reviving date",
    "tillering date",
    "jointing date",
    "booting date",
    "heading date",
]


def _inclusive_transition_sums(
    weather: pd.DataFrame,
    phenology: pd.Series,
    *,
    use_photoperiod: bool,
) -> list[float]:
    values: list[float] = []
    for stage_name, start_column, end_stage in zip(
        STAGE_NAMES,
        TRANSITION_START_COLUMNS,
        STAGE_NAMES,
    ):
        start = pd.to_datetime(phenology.get(start_column), errors="coerce")
        end = pd.to_datetime(phenology.get(f"{end_stage} date"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            values.append(float("nan"))
            continue

        interval = weather.loc[
            (weather["Date"] >= start) & (weather["Date"] <= end)
        ].copy()
        expected_days = int((end.normalize() - start.normalize()).days) + 1
        observed_dates = pd.to_datetime(interval["Date"]).dt.normalize()
        if (
            interval.empty
            or len(interval) != expected_days
            or observed_dates.nunique() != expected_days
        ):
            values.append(float("nan"))
            continue

        daily_development = interval["thermal"].to_numpy(dtype=float)
        if use_photoperiod and stage_name in PHOTO_SENSITIVE_STAGES:
            daily_development = (
                daily_development * interval["photo"].to_numpy(dtype=float)
            )
        requirement = float(np.sum(daily_development))
        values.append(requirement if requirement > 0.0 else float("nan"))
    return values


def _rollout_stage_signals(
    doy: np.ndarray,
    stage_signals: list[np.ndarray],
    start_doy: float,
    requirements: list[float],
) -> list[float]:
    predictions: list[float] = []
    current_start = float(start_doy)
    for signal, requirement in zip(stage_signals, requirements):
        if not np.isfinite(requirement) or requirement <= 0.0:
            predictions.extend([float("nan")] * (len(requirements) - len(predictions)))
            break
        eligible = np.flatnonzero(
            np.isfinite(doy) & np.isfinite(signal) & (doy >= current_start)
        )[:MAX_TRANSITION_DAYS]
        if not len(eligible):
            predictions.extend([float("nan")] * (len(requirements) - len(predictions)))
            break

        cumulative = np.cumsum(signal[eligible])
        crossings = np.flatnonzero(cumulative >= requirement)
        completion = float(
            doy[eligible[crossings[0]]] if len(crossings) else doy[eligible[-1]]
        )
        predictions.append(completion)
        current_start = completion + 1.0
    return predictions


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
        thermal = df["thermal"].to_numpy(dtype=float)
        photothermal = thermal * df["photo"].to_numpy(dtype=float)
        stage_signals = [
            photothermal if stage_name in PHOTO_SENSITIVE_STAGES else thermal
            for stage_name in STAGE_NAMES
        ]
        return _rollout_stage_signals(
            df["doy"].to_numpy(dtype=float),
            stage_signals,
            reviving_doy,
            thresholds,
        )

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
            if weather.empty:
                continue
            requirements = _inclusive_transition_sums(
                weather,
                row,
                use_photoperiod=True,
            )
            if not np.isfinite(requirements).any():
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
                    **dict(zip(THRESHOLD_COLUMNS, requirements)),
                }
            )
        return pd.DataFrame(rows)

    def fit(self, weather_df: pd.DataFrame, phenology_df: pd.DataFrame) -> dict[str, float]:
        threshold_df = self.collect_threshold_samples(weather_df, phenology_df)
        for column in THRESHOLD_COLUMNS:
            self.thresholds[column] = float(threshold_df[column].median())
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
    """Temperature-only baseline model without photoperiod effects.

    Key differences from M0PhenologyModel:
    - Uses only trapezoidal_temperature_response to calculate daily development contributions.
    - Uses factor = 1.0 for all stages, without photoperiod scaling for booting/heading.
    - Uses historical temperature-only accumulation for threshold inversion.
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
        """Compute thermal without computing photo."""
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
        """Run a temperature-only simulation with factor = 1.0 for all stages."""
        df = self._prepare_weather_t(weather_df)
        df["doy"] = df["Date"].dt.dayofyear
        thermal = df["thermal"].to_numpy(dtype=float)
        return _rollout_stage_signals(
            df["doy"].to_numpy(dtype=float),
            [thermal] * len(STAGE_NAMES),
            reviving_doy,
            thresholds,
        )

    def collect_threshold_samples_t(self, weather_df: pd.DataFrame, phenology_df: pd.DataFrame) -> pd.DataFrame:
        """Invert temperature-only thresholds using only historical thermal accumulation."""
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
            if weather.empty:
                continue
            requirements = _inclusive_transition_sums(
                weather,
                row,
                use_photoperiod=False,
            )
            if not np.isfinite(requirements).any():
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
                    **dict(zip(THRESHOLD_COLUMNS, requirements)),
                }
            )
        return pd.DataFrame(rows)

    def fit(self, weather_df: pd.DataFrame, phenology_df: pd.DataFrame) -> dict[str, float]:
        """Fit temperature-only thresholds."""
        threshold_df = self.collect_threshold_samples_t(weather_df, phenology_df)
        for column in THRESHOLD_COLUMNS:
            self.thresholds[column] = float(threshold_df[column].median())
        return dict(self.thresholds)

    def predict_one(self, weather_df: pd.DataFrame, sample: pd.Series) -> list[float]:
        thresholds = [self.thresholds[column] for column in THRESHOLD_COLUMNS]
        return self._simulate_stage_doys_t(
            weather_df=weather_df,
            reviving_doy=float(sample["obs_reviving"] if "obs_reviving" in sample else pd.to_datetime(sample["reviving date"]).dayofyear),
            thresholds=thresholds,
        )
