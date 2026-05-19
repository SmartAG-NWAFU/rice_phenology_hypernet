from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml

from rice_phenology_hypernet.settings import SETTINGS


ALLOWED_BASE_FEATURE_NAMES = (
    "latitude",
    "altitude",
    "transplanting_doy",
    "reviving_doy",
    "seeding_doy",
    "daylength_transplanting",
    "annual_mean_temperature",
    "annual_mean_tmax",
    "annual_mean_tmin",
    "annual_precipitation",
    "annual_mean_radiation",
    "gdd_gt_10",
)
WINDOW_FEATURE_PREFIXES = ("tmean", "paccum", "rmean", "hdd", "cdd")
SUPPORTED_EXPLICIT_FEATURE_NAMES = (
    "tmean_tran0_25",
    "tmean_tran0_50",
    "tmean_tran0_75",
    "tmean_tran0_100",
    "tmean_tran0_125",
    "tmean_tran30_60",
    "tmean_tran60_90",
    "tmean_tran90_120",
)
WINDOW_FEATURE_NAME_PATTERN = re.compile(rf"^(?:{'|'.join(WINDOW_FEATURE_PREFIXES)})_tran\d+_\d+$")
PUBLIC_MODEL_NAMES = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")


@dataclass(frozen=True)
class DataConfig:
    raw_weather: Path
    raw_phenology: Path
    china_boundary: Path
    province_boundary: Path


@dataclass(frozen=True)
class WindowConfig:
    label: str
    offset: int
    days: int


@dataclass(frozen=True)
class FeatureSelectionConfig:
    vif_threshold: float
    top_k_scatter: int


@dataclass(frozen=True)
class FeaturesConfig:
    temperature_base: float
    high_temp_threshold: float
    low_temp_threshold: float
    base_feature_names: tuple[str, ...]
    window_feature_families: tuple[str, ...]
    windows: tuple[WindowConfig, ...]
    explicit_feature_names: tuple[str, ...]
    generated_feature_names: tuple[str, ...]
    model_feature_sets: dict[str, tuple[str, ...]]
    default_model_feature_sets: dict[str, str]
    feature_selection: FeatureSelectionConfig


@dataclass(frozen=True)
class SampleConfig:
    n_splits: int
    seed: int
    min_test_samples: int


@dataclass(frozen=True)
class SiteConfig:
    n_splits: int
    min_test_samples: int
    min_test_sites: int


@dataclass(frozen=True)
class YearFoldConfig:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class YearConfig:
    folds: tuple[YearFoldConfig, ...]
    min_test_samples: int


@dataclass(frozen=True)
class DvrBatchConfig:
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class DvrCorrectionConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    early_stopping_patience: int
    inner_validation_strategy: str
    hidden_size: int
    dropout: float
    max_sequence_length: int
    grad_clip: float
    modifier_cap: float
    event_beta: float
    event_loss_weight: float
    terminal_loss_weight: float
    shrink_loss_weight: float
    smooth_loss_weight: float
    selection_metric: str
    mean_anchor_loss_weight: float
    stage_anchor_multipliers: tuple[float, ...]
    stage_terminal_weights: tuple[float, ...]
    stage_shrink_multipliers: tuple[float, ...]


@dataclass(frozen=True)
class ConstrainedDvrCorrectionConfig(DvrCorrectionConfig):
    background_gate_prior: tuple[float, ...]
    gate_prior_weight: float
    gate_monotonic_weight: float


@dataclass(frozen=True)
class ExperimentConfig:
    sample: SampleConfig
    site: SiteConfig
    year: YearConfig
    dvr_batch: DvrBatchConfig
    m1_v2_dvr: DvrCorrectionConfig
    m1_dvr_con: ConstrainedDvrCorrectionConfig


@dataclass(frozen=True)
class FigureStyleConfig:
    primary: str
    secondary: str
    accent: str


@dataclass(frozen=True)
class FigureConfig:
    dpi: int
    map_extent: tuple[float, float, float, float]
    style: FigureStyleConfig


@dataclass(frozen=True)
class LatestModelConfig:
    model: str
    model_label: str
    seeds: tuple[int, ...]
    model_options: dict[str, Any]


@dataclass(frozen=True)
class PaperConfig:
    comparison_models: tuple[str, ...]
    latest_model: LatestModelConfig


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    features: FeaturesConfig
    experiment: ExperimentConfig
    figures: FigureConfig
    paper: PaperConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload or {})


def _as_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _as_float_tuple(values: list[Any] | tuple[Any, ...], *, expected: int) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if len(resolved) != expected:
        raise ValueError(f"Expected {expected} values, got {len(resolved)}")
    return resolved


def _as_float_weights(values: list[Any] | tuple[Any, ...], *, field: str) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if len(resolved) != 5:
        raise ValueError(f"{field} must contain exactly five stage weights")
    return resolved


def _parse_data_config(root: Path, payload: dict[str, Any]) -> DataConfig:
    return DataConfig(
        raw_weather=_as_path(root, payload["raw_weather"]),
        raw_phenology=_as_path(root, payload["raw_phenology"]),
        china_boundary=_as_path(root, payload["china_boundary"]),
        province_boundary=_as_path(root, payload["province_boundary"]),
    )


def _validate_feature_name(name: str) -> None:
    if name in ALLOWED_BASE_FEATURE_NAMES:
        return
    if name in SUPPORTED_EXPLICIT_FEATURE_NAMES:
        return
    if WINDOW_FEATURE_NAME_PATTERN.match(name):
        return
    raise ValueError(f"Unsupported feature name in config: {name}")


def _generate_feature_names(payload: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = [str(value) for value in payload.get("base_feature_names", [])]
    for explicit in payload.get("explicit_feature_names", []) or []:
        names.append(str(explicit))
    families = tuple(str(value) for value in payload.get("window_feature_families", []) or [])
    for window in payload.get("windows", []) or []:
        label, _offset, _days = window
        for family in families:
            names.append(f"{family}_{label}")
    for name in names:
        _validate_feature_name(name)
    seen: set[str] = set()
    deduplicated: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            deduplicated.append(name)
    return tuple(deduplicated)


def _parse_features_config(payload: dict[str, Any]) -> FeaturesConfig:
    generated = _generate_feature_names(payload)
    raw_feature_sets = payload.get("model_feature_sets", {}) or {}
    feature_sets = {str(name): tuple(str(value) for value in values) for name, values in raw_feature_sets.items()}
    generated_set = set(generated)
    for set_name, values in feature_sets.items():
        missing = sorted(set(values) - generated_set)
        if missing:
            raise ValueError(f"Feature set {set_name!r} references undefined features: {missing}")
    return FeaturesConfig(
        temperature_base=float(payload["temperature_base"]),
        high_temp_threshold=float(payload["high_temp_threshold"]),
        low_temp_threshold=float(payload["low_temp_threshold"]),
        base_feature_names=tuple(str(value) for value in payload.get("base_feature_names", []) or []),
        window_feature_families=tuple(str(value) for value in payload.get("window_feature_families", []) or []),
        windows=tuple(WindowConfig(str(label), int(offset), int(days)) for label, offset, days in payload.get("windows", []) or []),
        explicit_feature_names=tuple(str(value) for value in payload.get("explicit_feature_names", []) or []),
        generated_feature_names=generated,
        model_feature_sets=feature_sets,
        default_model_feature_sets={
            str(name): str(value) for name, value in (payload.get("default_model_feature_sets", {}) or {}).items()
        },
        feature_selection=FeatureSelectionConfig(
            vif_threshold=float((payload.get("feature_selection", {}) or {}).get("vif_threshold", 10.0)),
            top_k_scatter=int((payload.get("feature_selection", {}) or {}).get("top_k_scatter", 2)),
        ),
    )


def _parse_sample_config(payload: dict[str, Any]) -> SampleConfig:
    return SampleConfig(
        n_splits=int(payload["n_splits"]),
        seed=int(payload["seed"]),
        min_test_samples=int(payload.get("min_test_samples", 1)),
    )


def _parse_site_config(payload: dict[str, Any]) -> SiteConfig:
    return SiteConfig(
        n_splits=int(payload["n_splits"]),
        min_test_samples=int(payload.get("min_test_samples", 1)),
        min_test_sites=int(payload.get("min_test_sites", 1)),
    )


def _parse_year_config(payload: dict[str, Any]) -> YearConfig:
    return YearConfig(
        folds=tuple(
            YearFoldConfig(
                fold=int(item["fold"]),
                train_start=int(item["train_start"]),
                train_end=int(item["train_end"]),
                test_start=int(item["test_start"]),
                test_end=int(item["test_end"]),
            )
            for item in payload.get("folds", []) or []
        ),
        min_test_samples=int(payload.get("min_test_samples", 1)),
    )


def _parse_dvr_config(payload: dict[str, Any]) -> DvrCorrectionConfig:
    return DvrCorrectionConfig(
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        validation_fraction=float(payload["validation_fraction"]),
        early_stopping_patience=int(payload["early_stopping_patience"]),
        inner_validation_strategy=str(payload.get("inner_validation_strategy", "protocol_aligned")),
        hidden_size=int(payload["hidden_size"]),
        dropout=float(payload["dropout"]),
        max_sequence_length=int(payload["max_sequence_length"]),
        grad_clip=float(payload["grad_clip"]),
        modifier_cap=float(payload["modifier_cap"]),
        event_beta=float(payload["event_beta"]),
        event_loss_weight=float(payload["event_loss_weight"]),
        terminal_loss_weight=float(payload["terminal_loss_weight"]),
        shrink_loss_weight=float(payload["shrink_loss_weight"]),
        smooth_loss_weight=float(payload["smooth_loss_weight"]),
        selection_metric=str(payload.get("selection_metric", "val_loader_loss")),
        mean_anchor_loss_weight=float(payload.get("mean_anchor_loss_weight", 0.0)),
        stage_anchor_multipliers=_as_float_weights(payload["stage_anchor_multipliers"], field="stage_anchor_multipliers"),
        stage_terminal_weights=_as_float_weights(payload["stage_terminal_weights"], field="stage_terminal_weights"),
        stage_shrink_multipliers=_as_float_weights(payload["stage_shrink_multipliers"], field="stage_shrink_multipliers"),
    )


def _parse_constrained_dvr_config(payload: dict[str, Any]) -> ConstrainedDvrCorrectionConfig:
    base = _parse_dvr_config(payload)
    return ConstrainedDvrCorrectionConfig(
        **base.__dict__,
        background_gate_prior=_as_float_weights(payload["background_gate_prior"], field="background_gate_prior"),
        gate_prior_weight=float(payload["gate_prior_weight"]),
        gate_monotonic_weight=float(payload["gate_monotonic_weight"]),
    )


def _parse_experiment_config(payload: dict[str, Any]) -> ExperimentConfig:
    seeds = tuple(int(value) for value in (payload.get("dvr_batch", {}) or {}).get("seeds", []))
    if not 1 <= len(seeds) <= 10:
        raise ValueError("DVR batch seeds must contain between 1 and 10 integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("DVR batch seeds must contain unique integers")
    return ExperimentConfig(
        sample=_parse_sample_config(payload["sample"]),
        site=_parse_site_config(payload["site"]),
        year=_parse_year_config(payload["year"]),
        dvr_batch=DvrBatchConfig(seeds=seeds),
        m1_v2_dvr=_parse_dvr_config(payload["m1_v2_dvr"]),
        m1_dvr_con=_parse_constrained_dvr_config(payload["m1_dvr_con"]),
    )


def _parse_figure_config(payload: dict[str, Any]) -> FigureConfig:
    style = payload.get("style", {}) or {}
    return FigureConfig(
        dpi=int(payload.get("dpi", 300)),
        map_extent=_as_float_tuple(payload.get("map_extent", [95, 125, 16, 45]), expected=4),
        style=FigureStyleConfig(
            primary=str(style.get("primary", "#2a9d8f")),
            secondary=str(style.get("secondary", "#e76f51")),
            accent=str(style.get("accent", "#264653")),
        ),
    )


def _parse_paper_config(payload: dict[str, Any]) -> PaperConfig:
    comparison_models = tuple(str(value) for value in payload.get("comparison_models", []) or [])
    if len(comparison_models) != len(set(comparison_models)):
        raise ValueError("paper.comparison_models must contain unique values")
    unsupported = sorted(set(comparison_models) - set(PUBLIC_MODEL_NAMES))
    if unsupported:
        raise ValueError(f"paper.comparison_models contains unsupported values: {unsupported}")
    latest = payload.get("latest_model", {}) or {}
    latest_model = str(latest.get("model", "m1_dvr_con"))
    if latest_model not in PUBLIC_MODEL_NAMES:
        raise ValueError(f"latest_model.model must be one of: {', '.join(PUBLIC_MODEL_NAMES)}")
    return PaperConfig(
        comparison_models=comparison_models,
        latest_model=LatestModelConfig(
            model=latest_model,
            model_label=str(latest.get("model_label", latest_model)),
            seeds=tuple(int(value) for value in latest.get("seeds", []) or []),
            model_options=dict(latest.get("model_options", {}) or {}),
        ),
    )


def load_project_config(root: Path | str = SETTINGS.root, configs_dir: Path | str | None = None) -> ProjectConfig:
    project_root = Path(root)
    resolved_configs_dir = Path(configs_dir) if configs_dir is not None else project_root / "configs"
    return ProjectConfig(
        data=_parse_data_config(project_root, _load_yaml(resolved_configs_dir / "data.yaml")),
        features=_parse_features_config(_load_yaml(resolved_configs_dir / "features.yaml")),
        experiment=_parse_experiment_config(_load_yaml(resolved_configs_dir / "experiment.yaml")),
        figures=_parse_figure_config(_load_yaml(resolved_configs_dir / "figures.yaml")),
        paper=_parse_paper_config(_load_yaml(resolved_configs_dir / "paper.yaml")),
    )


@lru_cache(maxsize=1)
def get_project_config() -> ProjectConfig:
    return load_project_config(SETTINGS.root, SETTINGS.configs_dir)


def get_generated_feature_names() -> tuple[str, ...]:
    return get_project_config().features.generated_feature_names


def get_model_feature_set(name: str) -> tuple[str, ...]:
    config = get_project_config()
    try:
        return config.features.model_feature_sets[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model feature set: {name}") from exc


def get_default_feature_set_for_model(model: str) -> tuple[str, ...]:
    config = get_project_config()
    set_name = config.features.default_model_feature_sets.get(model)
    if set_name is None:
        return config.features.generated_feature_names
    return get_model_feature_set(set_name)
