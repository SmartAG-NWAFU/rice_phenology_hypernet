from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rice_phenology_hypernet.data.daylength import DayLengthCalculator
from rice_phenology_hypernet.models.physics import oryza2000_photo_response, trapezoidal_temperature_response


DVR_STAGE_NAMES = ("tillering", "jointing", "booting", "heading", "maturity")
STAGE_END_COLUMNS = {
    "tillering": "obs_tillering",
    "jointing": "obs_jointing",
    "booting": "obs_booting",
    "heading": "obs_heading",
    "maturity": "obs_maturity",
}
PHOTO_SENSITIVE_STAGES = {"booting", "heading"}
DEFAULT_WEATHER_FEATURES = ("TemAver", "TemMin", "TemMax", "daylength", "Precipitation")
RADIATION_WEATHER_FEATURES = DEFAULT_WEATHER_FEATURES + ("Radiation",)


def _safe_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _finite_or(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else float(default)


def _stage_boundaries(row: pd.Series) -> list[tuple[int, str, int, int]]:
    reviving_doy = _safe_float(row.get("obs_reviving"))
    if not np.isfinite(reviving_doy):
        return []

    boundaries: list[tuple[int, str, int, int]] = []
    last_available_end = int(round(reviving_doy))
    for stage_index, stage_name in enumerate(DVR_STAGE_NAMES):
        end_doy = _safe_float(row.get(STAGE_END_COLUMNS[stage_name]))
        if not np.isfinite(end_doy):
            continue
        start_doy = int(round(reviving_doy)) if stage_index == 0 else last_available_end + 1
        end_doy_int = int(round(end_doy))
        if end_doy_int < start_doy:
            continue
        boundaries.append((stage_index, stage_name, start_doy, end_doy_int))
        last_available_end = end_doy_int
    return boundaries


def _ensure_daylength(weather: pd.DataFrame, latitude: float, calculator: DayLengthCalculator) -> pd.DataFrame:
    if "daylength" in weather.columns:
        return weather
    out = weather.copy()
    out["daylength"] = [calculator.day_length(d.year, d.month, d.day, latitude) for d in out["Date"]]
    return out


def _stage_factor(daylength: np.ndarray, stage_name: str) -> np.ndarray:
    if stage_name not in PHOTO_SENSITIVE_STAGES:
        return np.ones_like(daylength, dtype=np.float32)
    return np.asarray([oryza2000_photo_response(float(value)) for value in daylength], dtype=np.float32)


def _raw_stage_development(temperature: np.ndarray, daylength: np.ndarray, stage_name: str) -> np.ndarray:
    thermal = trapezoidal_temperature_response(temperature).astype(np.float32)
    factor = _stage_factor(daylength.astype(np.float32), stage_name)
    return thermal * factor


def _raw_stage_development_t(temperature: np.ndarray) -> np.ndarray:
    """温度-only 版本：不考虑光周期效应，所有阶段统一使用 factor=1。"""
    thermal = trapezoidal_temperature_response(temperature).astype(np.float32)
    return thermal


def _compute_gdd_history(
    weather: pd.DataFrame,
    transplanting_doy: int,
    start_doy: int,
    latitude: float,
    daylength_calculator: DayLengthCalculator,
) -> float:
    """计算从 transplanting_doy 到 start_doy 的历史光温累计。

    使用 m0 的分段规则：
    - 需要知道每个阶段的边界才能正确分段
    - 但对于 stage sample，我们只需要计算"之前"的历史
    - 简化处理：直接使用温度响应（不考虑光敏），因为历史段可能跨越多个阶段

    这里计算的是"从移栽到当前阶段开始"的累积发育势，使用统一的温度响应
    （不分段），因为分段需要知道阶段边界。
    """
    if start_doy <= transplanting_doy:
        return 0.0

    # 提取历史天气数据
    history_weather = weather[
        (weather["doy"] >= transplanting_doy) & (weather["doy"] < start_doy)
    ].copy()

    if history_weather.empty:
        return 0.0

    # 简化处理：使用温度响应累计（不分段）
    # 分段需要知道各阶段边界，但当前只有 transplanting_doy 和 start_doy
    thermal = trapezoidal_temperature_response(history_weather["TemAver"].to_numpy(dtype=np.float32))
    return float(thermal.sum())


def estimate_median_stage_start_gdd(samples: list[dict[str, object]]) -> dict[str, float]:
    """估计每个阶段开始时的 median gdd_history。

    用于归一化：gdd_history_rel = gdd_history / median_stage_start_gdd[stage_index]
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        value = float(sample.get("gdd_history", 0.0))
        if np.isfinite(value):
            grouped[str(sample["stage_name"])].append(value)
    return {stage_name: float(np.median(values)) if values else 1.0 for stage_name, values in grouped.items()}


class RiceDvrStageDataset(Dataset):
    def __init__(
        self,
        modeling_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        *,
        stage_requirements: dict[str, float] | None = None,
        stage_rate_priors: dict[str, float] | None = None,
        weather_features: tuple[str, ...] = DEFAULT_WEATHER_FEATURES,
        median_stage_start_gdd: dict[str, float] | None = None,
    ):
        self.modeling_df = modeling_df.reset_index(drop=True).copy()
        self.weather_index = {
            (int(sid), int(year)): group.sort_values("Date").reset_index(drop=True)
            for (sid, year), group in weather_df.groupby(["SID", "year"])
        }
        self.stage_requirements = dict(stage_requirements or {})
        self.stage_rate_priors = dict(stage_rate_priors or {})
        self.weather_features = tuple(weather_features)
        self.median_stage_start_gdd = dict(median_stage_start_gdd or {})
        self.daylength = DayLengthCalculator()
        self.samples = [sample for sample in self._build_samples() if sample is not None]

    def _build_samples(self) -> list[dict[str, object] | None]:
        samples: list[dict[str, object] | None] = []
        for _, row in self.modeling_df.iterrows():
            weather = self.weather_index.get((int(row["SID"]), int(row["year"])))
            if weather is None or weather.empty:
                continue
            latitude = float(row["latitude"])
            season_weather = weather[weather["Date"].dt.year == int(row["year"])].copy()
            season_weather = _ensure_daylength(season_weather, latitude, self.daylength)
            if season_weather.empty:
                continue
            season_weather["doy"] = season_weather["Date"].dt.dayofyear.astype(int)

            for stage_index, stage_name, start_doy, end_doy in _stage_boundaries(row):
                seq = season_weather[(season_weather["doy"] >= start_doy) & (season_weather["doy"] <= end_doy)].copy()
                if seq.empty:
                    continue
                weather_seq = seq.loc[:, self.weather_features].to_numpy(dtype=np.float32)
                daylength = seq["daylength"].to_numpy(dtype=np.float32)
                raw_base_seq = _raw_stage_development(seq["TemAver"].to_numpy(dtype=np.float32), daylength, stage_name)
                raw_requirement = float(raw_base_seq.sum())
                if not np.isfinite(raw_requirement) or raw_requirement <= 0:
                    continue
                base_requirement = float(self.stage_requirements.get(stage_name, raw_requirement))
                transplanting_doy = _finite_or(_safe_float(row.get("transplanting_doy")), float(start_doy))

                # 计算 gdd_history（从移栽到当前阶段开始的历史光温累计）
                gdd_history = _compute_gdd_history(
                    season_weather, int(transplanting_doy), start_doy, latitude, self.daylength
                )
                # 归一化
                median_gdd = self.median_stage_start_gdd.get(stage_name, 1.0)
                gdd_history_rel = gdd_history / max(median_gdd, 1e-6)

                sample = {
                    "sid": int(row["SID"]),
                    "year": int(row["year"]),
                    "stage_index": int(stage_index),
                    "stage_name": stage_name,
                    "start_doy": int(start_doy),
                    "end_doy": int(end_doy),
                    "true_duration": int(len(seq)),
                    "weather_seq": torch.tensor(weather_seq, dtype=torch.float32),
                    "base_dvr_seq": torch.tensor(raw_base_seq / base_requirement, dtype=torch.float32),
                    "mask": torch.ones(len(seq), dtype=torch.bool),
                    "stage_state": torch.tensor(
                        [
                            float(start_doy),
                            float(start_doy) - transplanting_doy,
                        ],
                        dtype=torch.float32,
                    ),
                    "gdd_history": float(gdd_history),
                    "gdd_history_rel": torch.tensor([gdd_history_rel], dtype=torch.float32),
                    "base_requirement": float(base_requirement),
                    "raw_base_requirement": float(raw_requirement),
                    "stage_rate_prior": float(self.stage_rate_priors.get(stage_name, 1.0 / max(len(seq), 1))),
                    "transplanting_doy": float(transplanting_doy),
                }
                samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.samples[idx]


def estimate_stage_requirements(samples: list[dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        value = float(sample["raw_base_requirement"])
        if np.isfinite(value) and value > 0:
            grouped[str(sample["stage_name"])].append(value)
    return {stage_name: float(np.median(values)) for stage_name, values in grouped.items() if values}


def estimate_stage_requirements_t(
    modeling_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> dict[str, float]:
    """温度-only 版本的阶段阈值估计。

    直接遍历 modeling_df 和 weather_df，使用温度-only 的日贡献计算每个样本的阶段阈值，
    然后取各阶段的 median。

    与 estimate_stage_requirements() 的关键差异：
    - 使用 _raw_stage_development_t() 而不是 _raw_stage_development()
    - 不需要创建 RiceDvrStageDataset，直接处理原始数据
    """
    weather_index = {
        (int(sid), int(year)): group.sort_values("Date").reset_index(drop=True)
        for (sid, year), group in weather_df.groupby(["SID", "year"])
    }
    daylength_calculator = DayLengthCalculator()
    grouped: dict[str, list[float]] = defaultdict(list)

    for _, row in modeling_df.iterrows():
        key = (int(row["SID"]), int(row["year"]))
        if key not in weather_index:
            continue
        weather = weather_index[key]
        latitude = float(row["latitude"])
        season_weather = weather[weather["Date"].dt.year == int(row["year"])].copy()
        season_weather = _ensure_daylength(season_weather, latitude, daylength_calculator)
        if season_weather.empty:
            continue
        season_weather["doy"] = season_weather["Date"].dt.dayofyear.astype(int)

        for stage_index, stage_name, start_doy, end_doy in _stage_boundaries(row):
            seq = season_weather[(season_weather["doy"] >= start_doy) & (season_weather["doy"] <= end_doy)].copy()
            if seq.empty:
                continue
            # 温度-only：使用 _raw_stage_development_t()
            raw_base_seq = _raw_stage_development_t(seq["TemAver"].to_numpy(dtype=np.float32))
            raw_requirement = float(raw_base_seq.sum())
            if np.isfinite(raw_requirement) and raw_requirement > 0:
                grouped[stage_name].append(raw_requirement)

    return {stage_name: float(np.median(values)) for stage_name, values in grouped.items() if values}


def estimate_stage_rate_priors(samples: list[dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        duration = pd.to_numeric(pd.Series([sample.get("true_duration")]), errors="coerce").iloc[0]
        if pd.notna(duration) and float(duration) > 0:
            grouped[str(sample["stage_name"])].append(float(duration))
    priors: dict[str, float] = {}
    for stage_name, durations in grouped.items():
        median_duration = float(np.median(durations))
        if median_duration > 0:
            priors[stage_name] = 1.0 / median_duration
    return priors


class RiceDvrSeasonDataset(Dataset):
    def __init__(
        self,
        modeling_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        *,
        weather_features: tuple[str, ...] = DEFAULT_WEATHER_FEATURES,
        max_sequence_length: int = 130,
    ):
        self.modeling_df = modeling_df.reset_index(drop=True).copy()
        self.weather_index = {
            (int(sid), int(year)): group.sort_values("Date").reset_index(drop=True)
            for (sid, year), group in weather_df.groupby(["SID", "year"])
        }
        self.weather_features = tuple(weather_features)
        self.max_sequence_length = int(max_sequence_length)
        self.daylength = DayLengthCalculator()
        self.samples = [sample for sample in self._build_samples() if sample is not None]

    def _build_samples(self) -> list[dict[str, object] | None]:
        samples: list[dict[str, object] | None] = []
        for _, row in self.modeling_df.iterrows():
            weather = self.weather_index.get((int(row["SID"]), int(row["year"])))
            if weather is None or weather.empty:
                continue
            reviving_doy = _safe_float(row.get("obs_reviving"))
            if not np.isfinite(reviving_doy):
                continue
            latitude = float(row["latitude"])
            season_weather = weather[weather["Date"].dt.year == int(row["year"])].copy()
            season_weather = _ensure_daylength(season_weather, latitude, self.daylength)
            if season_weather.empty:
                continue
            season_weather["doy"] = season_weather["Date"].dt.dayofyear.astype(int)
            seq = season_weather[season_weather["doy"] >= int(round(reviving_doy))].head(self.max_sequence_length).copy()
            if seq.empty:
                continue

            weather_seq = np.zeros((self.max_sequence_length, len(self.weather_features)), dtype=np.float32)
            seq_values = seq.loc[:, self.weather_features].to_numpy(dtype=np.float32)
            weather_seq[: len(seq_values)] = seq_values
            mask = np.zeros(self.max_sequence_length, dtype=bool)
            mask[: len(seq_values)] = True

            target_dates = []
            target_durations = []
            sample_record: dict[str, object] = {
                "sid": int(row["SID"]),
                "year": int(row["year"]),
                "start_doy": float(reviving_doy),
                "weather_seq": torch.tensor(weather_seq, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "season_state": torch.tensor([float(reviving_doy)], dtype=torch.float32),
            }

            for stage_index, stage_name in enumerate(DVR_STAGE_NAMES):
                end_doy = _safe_float(row.get(STAGE_END_COLUMNS[stage_name]))
                target_dates.append(end_doy)
                if stage_index == 0:
                    prev_stage_end = float(reviving_doy) - 1.0
                else:
                    prev_stage_name = DVR_STAGE_NAMES[stage_index - 1]
                    prev_stage_end = _safe_float(row.get(STAGE_END_COLUMNS[prev_stage_name]))
                duration = end_doy - prev_stage_end if np.isfinite(end_doy) and np.isfinite(prev_stage_end) else float("nan")
                sample_record[f"target_duration_{stage_name}"] = float(duration) if np.isfinite(duration) else float("nan")
                target_durations.append(float(duration) if np.isfinite(duration) else float("nan"))

            sample_record["target_dates"] = torch.tensor(target_dates, dtype=torch.float32)
            sample_record["target_durations"] = torch.tensor(target_durations, dtype=torch.float32)
            samples.append(sample_record)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.samples[idx]

def collate_dvr_batches(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("DVR batch must contain at least one sample.")

    max_len = max(int(sample["true_duration"]) for sample in batch)
    feature_dim = int(batch[0]["weather_seq"].shape[1])

    weather_seq = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32)
    base_dvr_seq = torch.zeros((len(batch), max_len), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for idx, sample in enumerate(batch):
        length = int(sample["true_duration"])
        weather_seq[idx, :length] = sample["weather_seq"]
        base_dvr_seq[idx, :length] = sample["base_dvr_seq"]
        mask[idx, :length] = True

    # 检查是否有历史热量累计字段。
    has_gdd_history_rel = "gdd_history_rel" in batch[0]

    result = {
        "sid": torch.tensor([int(sample["sid"]) for sample in batch], dtype=torch.long),
        "year": torch.tensor([int(sample["year"]) for sample in batch], dtype=torch.long),
        "stage_index": torch.tensor([int(sample["stage_index"]) for sample in batch], dtype=torch.long),
        "stage_name": [str(sample["stage_name"]) for sample in batch],
        "start_doy": torch.tensor([int(sample["start_doy"]) for sample in batch], dtype=torch.long),
        "end_doy": torch.tensor([int(sample["end_doy"]) for sample in batch], dtype=torch.long),
        "true_duration": torch.tensor([int(sample["true_duration"]) for sample in batch], dtype=torch.long),
        "stage_state": torch.stack([sample["stage_state"] for sample in batch], dim=0),
        "weather_seq": weather_seq,
        "base_dvr_seq": base_dvr_seq,
        "mask": mask,
        "base_requirement": torch.tensor([float(sample["base_requirement"]) for sample in batch], dtype=torch.float32),
        "stage_rate_prior": torch.tensor([float(sample.get("stage_rate_prior", 0.0)) for sample in batch], dtype=torch.float32),
    }

    # 添加历史热量累计字段。
    if has_gdd_history_rel:
        result["gdd_history_rel"] = torch.stack([sample["gdd_history_rel"] for sample in batch], dim=0)
        result["gdd_history"] = torch.tensor([float(sample.get("gdd_history", 0.0)) for sample in batch], dtype=torch.float32)
        result["transplanting_doy"] = torch.tensor([float(sample.get("transplanting_doy", 0.0)) for sample in batch], dtype=torch.float32)

    return result
