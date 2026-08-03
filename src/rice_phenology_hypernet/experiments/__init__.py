"""Public four-model experiment interfaces."""

from .dvr_core import PAPER_MODEL_NAMES

__all__ = [
    "DvrExperimentBundle",
    "ExperimentSpec",
    "PAPER_MODEL_NAMES",
    "run_dvr_experiment",
]


def __getattr__(name: str):
    if name in {"DvrExperimentBundle", "ExperimentSpec", "run_dvr_experiment"}:
        from .runner_dvr import (
            DvrExperimentBundle,
            ExperimentSpec,
            run_dvr_experiment,
        )

        return {
            "DvrExperimentBundle": DvrExperimentBundle,
            "ExperimentSpec": ExperimentSpec,
            "run_dvr_experiment": run_dvr_experiment,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
