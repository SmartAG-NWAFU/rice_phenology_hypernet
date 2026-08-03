"""Compact orchestration for the four DVR models reported in the paper.

The runner owns the scientific sequence shared by every model: calculate a
daily process rate, apply an optional positive modifier, accumulate corrected
development, identify the first threshold crossing, and advance to the next
stage. Data splitting, feature construction, model fitting, and metric
calculation are supplied through an explicit backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from .dvr_core import (
    DVR_STAGE_NAMES,
    PAPER_MODEL_NAMES,
    StageInputs,
    StageRolloutResult,
)


@dataclass(frozen=True)
class ExperimentSpec:
    """Identity of one evaluation task and its sole random seed."""

    task: str
    model_name: str
    seed: int


@dataclass(frozen=True)
class FoldRecords:
    """Training and test records for one evaluation fold."""

    fold: int
    train_records: pd.DataFrame
    test_records: pd.DataFrame


@dataclass(frozen=True)
class DvrExperimentBundle:
    """Predictions, summary metrics, and provenance for one experiment."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    audit: dict[str, object]


class DvrLossSettings(Protocol):
    """Configuration-owned common objective settings."""

    event_loss_weight: float
    terminal_loss_weight: float
    shrink_loss_weight: float
    smooth_loss_weight: float
    mean_anchor_loss_weight: float
    stage_anchor_multipliers: tuple[float, ...]
    stage_terminal_weights: tuple[float, ...]
    stage_shrink_multipliers: tuple[float, ...]
    eps: float


class M1V2Settings(DvrLossSettings, Protocol):
    """Configuration-owned settings for M1-V2-DVR."""

    hidden_size: int
    dropout: float
    modifier_cap: float
    event_beta: float


class M1ConSettings(DvrLossSettings, Protocol):
    """Configuration-owned settings for M1-DVR-CON."""

    hidden_size: int
    dropout: float
    modifier_cap: float
    event_beta: float
    background_gate_prior: tuple[float, ...]
    gate_prior_weight: float
    gate_monotonic_weight: float


class DvrExperimentConfig(Protocol):
    """Injected records and learned-model configuration sections."""

    records: pd.DataFrame
    m1_v2_dvr: M1V2Settings
    m1_dvr_con: M1ConSettings


class DvrWorkflowBackend(Protocol):
    """Project-specific operations surrounding the shared DVR rollout."""

    def split_records(
        self,
        records: pd.DataFrame,
        *,
        task: str,
        seed: int,
    ) -> tuple[FoldRecords, ...]: ...

    def estimate_stage_requirements(
        self,
        train_records: pd.DataFrame,
    ) -> Mapping[str, float]: ...

    def fit_learned_model(
        self,
        model_name: str,
        model: object,
        train_records: pd.DataFrame,
        requirements: Mapping[str, float],
        config: object,
        *,
        seed: int,
    ) -> object: ...

    def build_stage_inputs(
        self,
        record: pd.Series,
        stage_name: str,
        stage_start_doy: float,
        requirements: Mapping[str, float],
    ) -> StageInputs: ...

    def calculate_base_dvr(
        self,
        model_name: str,
        stage_name: str,
        inputs: StageInputs,
        requirements: Mapping[str, float],
    ) -> np.ndarray: ...

    def calculate_modifier(
        self,
        model_name: str,
        model: object,
        stage_name: str,
        inputs: StageInputs,
        requirements: Mapping[str, float],
    ) -> np.ndarray | None: ...

    def build_prediction_record(
        self,
        record: pd.Series,
        stage_predictions: Mapping[str, float],
        *,
        fold: int,
        model_name: str,
    ) -> dict[str, object]: ...

    def score(self, predictions: pd.DataFrame) -> pd.DataFrame: ...


def _learned_config(
    model_name: str,
    config: DvrExperimentConfig,
) -> M1V2Settings | M1ConSettings | None:
    if model_name == "m1_v2_dvr":
        return config.m1_v2_dvr
    if model_name == "m1_dvr_con":
        return config.m1_dvr_con
    return None


def build_paper_model(
    model_name: str,
    config: DvrExperimentConfig,
) -> object:
    """Construct exactly one of the four paper models from injected settings."""

    if model_name not in PAPER_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected one of {PAPER_MODEL_NAMES!r}"
        )

    if model_name == "m0_t":
        from rice_phenology_hypernet.models.m0 import M0TPhenologyModel

        return M0TPhenologyModel()
    if model_name == "m0_dvr":
        from rice_phenology_hypernet.models.m0 import M0PhenologyModel

        return M0PhenologyModel()
    if model_name == "m1_v2_dvr":
        from rice_phenology_hypernet.models.m1_v2_dvr import (
            M1V2DvrConfig,
            M1V2DvrModel,
        )

        section = config.m1_v2_dvr
        return M1V2DvrModel(
            M1V2DvrConfig(
                hidden_size=section.hidden_size,
                dropout=section.dropout,
                modifier_cap=section.modifier_cap,
                event_beta=section.event_beta,
            )
        )

    from rice_phenology_hypernet.models.m1_dvr_con import (
        M1ConDvrConfig,
        M1ConDvrModel,
    )

    section = config.m1_dvr_con
    return M1ConDvrModel(
        M1ConDvrConfig(
            hidden_size=section.hidden_size,
            dropout=section.dropout,
            modifier_cap=section.modifier_cap,
            event_beta=section.event_beta,
            background_gate_prior=section.background_gate_prior,
        )
    )


def rollout_stage(
    inputs: StageInputs,
    base_dvr: np.ndarray,
    requirement: float,
    modifier: np.ndarray | None,
    stage_start_doy: float,
    *,
    trace: list[str] | None = None,
) -> StageRolloutResult:
    """Apply correction and return the first daily threshold crossing."""

    doy = np.asarray(inputs.doy, dtype=float)
    mask = np.asarray(inputs.mask, dtype=bool)
    base = np.asarray(base_dvr, dtype=float)
    if doy.ndim != 1 or mask.ndim != 1 or base.ndim != 1:
        raise ValueError("Stage DOY, mask, and base DVR inputs must be one-dimensional")
    if not (doy.shape == mask.shape == base.shape):
        raise ValueError("Stage DOY, mask, and base DVR inputs must share one shape")
    if not np.isfinite(requirement) or requirement <= 0:
        raise ValueError("The stage requirement must be a finite positive value")
    if not np.isfinite(stage_start_doy):
        raise ValueError("The stage start DOY must be finite")

    if modifier is None:
        effective_modifier = np.ones_like(base)
    else:
        effective_modifier = np.asarray(modifier, dtype=float)
        if effective_modifier.shape != base.shape:
            raise ValueError("The daily modifier must match the base DVR shape")

    if trace is not None:
        trace.append("correct")
    corrected = np.where(mask, base * effective_modifier, 0.0)
    if trace is not None:
        trace.append("accumulate")
    cumulative = np.cumsum(corrected)
    if trace is not None:
        trace.append("cross")
    crossings = np.flatnonzero(mask & (cumulative >= requirement))
    completion = float(doy[crossings[0]]) if len(crossings) else float("nan")
    if trace is not None:
        trace.append("advance")
    next_start = completion + 1.0 if np.isfinite(completion) else float("nan")
    return StageRolloutResult(
        completion_doy=completion,
        next_start_doy=next_start,
        corrected_dvr=corrected,
        cumulative_progress=cumulative,
    )


def _initial_stage_start_doy(record: pd.Series) -> float:
    for column in ("obs_reviving", "reviving_doy"):
        if column in record and pd.notna(record[column]):
            return float(record[column])
    if "reviving date" in record and pd.notna(record["reviving date"]):
        return float(pd.Timestamp(record["reviving date"]).dayofyear)
    raise ValueError(
        "Each record must provide obs_reviving, reviving_doy, or reviving date"
    )


def run_dvr_experiment(
    spec: ExperimentSpec,
    config: DvrExperimentConfig,
    backend: DvrWorkflowBackend,
) -> DvrExperimentBundle:
    """Run fold-isolated, sequential prediction for one paper model."""

    if spec.model_name not in PAPER_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model {spec.model_name!r}; expected one of {PAPER_MODEL_NAMES!r}"
        )

    call_order: list[str] = ["split"]
    folds = backend.split_records(config.records, task=spec.task, seed=spec.seed)
    prediction_rows: list[dict[str, object]] = []
    fold_audit: list[dict[str, object]] = []

    for fold in folds:
        call_order.append("estimate")
        requirements = backend.estimate_stage_requirements(fold.train_records)
        missing_stages = [stage for stage in DVR_STAGE_NAMES if stage not in requirements]
        if missing_stages:
            raise ValueError(
                f"Stage requirements are missing: {', '.join(missing_stages)}"
            )

        model = build_paper_model(spec.model_name, config)
        learned_config = _learned_config(spec.model_name, config)
        if learned_config is not None:
            call_order.append("fit")
            model = backend.fit_learned_model(
                spec.model_name,
                model,
                fold.train_records,
                requirements,
                learned_config,
                seed=spec.seed,
            )

        fold_trace: list[str] = []
        for _, record in fold.test_records.iterrows():
            stage_start_doy = _initial_stage_start_doy(record)
            stage_predictions: dict[str, float] = {}
            for stage_name in DVR_STAGE_NAMES:
                call_order.append("stage_inputs")
                inputs = backend.build_stage_inputs(
                    record,
                    stage_name,
                    stage_start_doy,
                    requirements,
                )
                call_order.append("base_dvr")
                base_dvr = backend.calculate_base_dvr(
                    spec.model_name,
                    stage_name,
                    inputs,
                    requirements,
                )
                call_order.append("modifier")
                modifier = backend.calculate_modifier(
                    spec.model_name,
                    model,
                    stage_name,
                    inputs,
                    requirements,
                )
                stage_requirement = float(requirements[stage_name])
                result = rollout_stage(
                    inputs,
                    base_dvr,
                    stage_requirement,
                    modifier,
                    stage_start_doy,
                    trace=call_order,
                )
                fold_trace.extend(call_order[-4:])
                stage_predictions[stage_name] = result.completion_doy
                stage_start_doy = result.next_start_doy
                if not np.isfinite(stage_start_doy):
                    break

            prediction_rows.append(
                backend.build_prediction_record(
                    record,
                    stage_predictions,
                    fold=fold.fold,
                    model_name=spec.model_name,
                )
            )

        fold_audit.append(
            {
                "fold": fold.fold,
                "requirement_object_id": id(requirements),
                "requirement_source": "training_records_only",
                "stage_trace": tuple(fold_trace),
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    call_order.append("score")
    metrics = backend.score(predictions)
    return DvrExperimentBundle(
        predictions=predictions,
        metrics=metrics,
        audit={
            "task": spec.task,
            "model_name": spec.model_name,
            "seed": spec.seed,
            "call_order": tuple(call_order),
            "folds": tuple(fold_audit),
        },
    )


def validate_recording_backend_contract() -> None:
    """Run a dependency-light synthetic check of ordering and object identity."""

    events: list[str] = []
    requirement_ids: list[int] = []

    class Section:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"Synthetic model construction accessed {name}")

    class Config:
        records = pd.DataFrame([{"reviving_doy": 1.0}])
        m1_v2_dvr = Section()
        m1_dvr_con = Section()

    class Backend:
        def split_records(self, records, *, task, seed):
            events.append("split")
            return (FoldRecords(0, records.copy(), records.copy()),)

        def estimate_stage_requirements(self, train_records):
            events.append("estimate")
            return {stage: 1.0 for stage in DVR_STAGE_NAMES}

        def fit_learned_model(
            self,
            model_name,
            model,
            train_records,
            requirements,
            config,
            *,
            seed,
        ):
            events.append("fit")
            requirement_ids.append(id(requirements))
            return model

        def build_stage_inputs(
            self,
            record,
            stage_name,
            stage_start_doy,
            requirements,
        ):
            events.append("stage_inputs")
            requirement_ids.append(id(requirements))
            return StageInputs(
                doy=np.array([stage_start_doy]),
                mask=np.array([True]),
                model_inputs=None,
            )

        def calculate_base_dvr(
            self,
            model_name,
            stage_name,
            inputs,
            requirements,
        ):
            events.append("base_dvr")
            requirement_ids.append(id(requirements))
            return np.ones_like(inputs.doy, dtype=float)

        def calculate_modifier(
            self,
            model_name,
            model,
            stage_name,
            inputs,
            requirements,
        ):
            events.append("modifier")
            requirement_ids.append(id(requirements))
            return None

        def build_prediction_record(
            self,
            record,
            stage_predictions,
            *,
            fold,
            model_name,
        ):
            return {"fold": fold, "model": model_name, **stage_predictions}

        def score(self, predictions):
            events.append("score")
            return pd.DataFrame([{"records": len(predictions)}])

    original_builder = globals()["build_paper_model"]
    globals()["build_paper_model"] = lambda model_name, config: object()
    try:
        bundle = run_dvr_experiment(
            ExperimentSpec(
                task="synthetic",
                model_name="m1_v2_dvr",
                seed=len(events),
            ),
            Config(),
            Backend(),
        )
    finally:
        globals()["build_paper_model"] = original_builder

    stage_events = ["stage_inputs", "base_dvr", "modifier"] * len(DVR_STAGE_NAMES)
    expected_backend_events = ["split", "estimate", "fit", *stage_events, "score"]
    if events != expected_backend_events:
        raise AssertionError(f"Unexpected backend event order: {events!r}")
    if len(set(requirement_ids)) != 1:
        raise AssertionError("The same stage-requirement object was not reused")
    expected_stage_trace = ("correct", "accumulate", "cross", "advance") * len(
        DVR_STAGE_NAMES
    )
    actual_trace = bundle.audit["folds"][0]["stage_trace"]
    if actual_trace != expected_stage_trace:
        raise AssertionError(f"Unexpected rollout trace: {actual_trace!r}")


__all__ = [
    "DvrExperimentBundle",
    "DvrExperimentConfig",
    "DvrWorkflowBackend",
    "ExperimentSpec",
    "FoldRecords",
    "PAPER_MODEL_NAMES",
    "build_paper_model",
    "rollout_stage",
    "run_dvr_experiment",
    "validate_recording_backend_contract",
]
