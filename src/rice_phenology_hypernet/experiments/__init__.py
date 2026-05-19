__all__ = [
    "analyze_modifier_interpretability",
    "run_all_dvr_experiments",
    "run_dvr_diagnostic",
    "run_dvr_experiment",
    "run_regional_reviving_offset_sensitivity",
    "train_dvr_deployment_models",
]


def __getattr__(name: str):
    if name == "analyze_modifier_interpretability":
        from .modifier_interpretability import analyze_modifier_interpretability

        return analyze_modifier_interpretability
    if name in {"run_all_dvr_experiments", "run_dvr_experiment", "train_dvr_deployment_models"}:
        from .runner_dvr import run_all_dvr_experiments, run_dvr_experiment, train_dvr_deployment_models

        return {
            "run_all_dvr_experiments": run_all_dvr_experiments,
            "run_dvr_experiment": run_dvr_experiment,
            "train_dvr_deployment_models": train_dvr_deployment_models,
        }[name]
    if name == "run_regional_reviving_offset_sensitivity":
        from .regional_reviving_offset_sensitivity import run_regional_reviving_offset_sensitivity

        return run_regional_reviving_offset_sensitivity
    if name == "run_dvr_diagnostic":
        from .dvr_diagnostic import run_dvr_diagnostic

        return run_dvr_diagnostic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
