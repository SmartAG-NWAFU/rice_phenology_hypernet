from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


PAPER_TASKS = ("sample", "site", "year")
PAPER_MODELS = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")
REGIONAL_PERIODS = ("2003_2007", "2008_2012", "2013_2017", "2018_2022", "2003_2022", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rice phenology reproducibility CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data")
    sub.add_parser("invert-thresholds")

    run_dvr_parser = sub.add_parser("run-dvr-experiment")
    run_dvr_parser.add_argument("--task", required=True, choices=PAPER_TASKS)
    run_dvr_parser.add_argument("--model", required=True, choices=PAPER_MODELS)
    run_dvr_parser.add_argument("--run-id")
    run_dvr_parser.add_argument("--seed", type=int)

    all_dvr_parser = sub.add_parser("run-all-dvr-experiments")
    all_dvr_parser.add_argument("--run-id")
    all_dvr_parser.add_argument("--seeds", nargs="+", type=int, default=None)
    all_dvr_parser.add_argument("--num-workers", type=int, default=None)
    all_dvr_parser.add_argument("--threads-per-worker", type=int, default=1)
    all_dvr_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    all_dvr_parser.add_argument("--gpu-workers", type=int, default=1)

    deploy_dvr_parser = sub.add_parser("train-dvr-deployment-models")
    deploy_dvr_parser.add_argument("--run-id")
    deploy_dvr_parser.add_argument("--seed", type=int, default=61)

    diagnostic_dvr_parser = sub.add_parser("run-dvr-diagnostic")
    diagnostic_dvr_parser.add_argument("--task", required=True, choices=PAPER_TASKS)
    diagnostic_dvr_parser.add_argument("--run-id")

    modifier_parser = sub.add_parser("analyze-modifier-interpretability")
    modifier_parser.add_argument("--deployment-run-id", type=str, default="molde4_seed61")
    modifier_parser.add_argument("--run-id")
    modifier_parser.add_argument("--seed", type=int, default=61)
    modifier_parser.add_argument("--stage", choices=["tillering", "jointing", "booting", "heading", "maturity"])

    regional_inputs_parser = sub.add_parser("prepare-regional-grid-inputs")
    regional_inputs_parser.add_argument("--period", choices=REGIONAL_PERIODS, default="2003_2022")
    regional_inputs_parser.add_argument("--remote-sensing-path", type=str, default=None)
    regional_inputs_parser.add_argument("--weather-summary-path", type=str, default=None)
    regional_inputs_parser.add_argument("--output-dir", type=str, default=None)

    regional_projection_parser = sub.add_parser("run-regional-grid-projection")
    regional_projection_parser.add_argument("--deployment-run-id", type=str, default="molde4_seed61")
    regional_projection_parser.add_argument("--run-id")
    regional_projection_parser.add_argument("--seed", type=int, default=61)
    regional_projection_parser.add_argument("--period", choices=REGIONAL_PERIODS, default="2003_2022")
    regional_projection_parser.add_argument("--input-path", type=str, default=None)
    regional_projection_parser.add_argument("--weather-dir", type=str, default=None)
    regional_projection_parser.add_argument("--chunk-size", type=int, default=2048)
    regional_projection_parser.add_argument("--num-workers", type=int, default=None)
    regional_projection_parser.add_argument("--threads-per-worker", type=int, default=1)
    regional_projection_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    regional_projection_parser.add_argument("--output-dir", type=str, default=None)

    regional_analysis_parser = sub.add_parser("analyze-regional-grid-projection")
    regional_analysis_parser.add_argument("--run-id")
    regional_analysis_parser.add_argument("--period", choices=REGIONAL_PERIODS, default="2003_2022")
    regional_analysis_parser.add_argument("--input-dir", type=str, default=None)
    regional_analysis_parser.add_argument("--yearly-predictions-path", type=str, default=None)
    regional_analysis_parser.add_argument("--output-dir", type=str, default=None)
    regional_analysis_parser.add_argument("--figures-dir", type=str, default=None)
    regional_analysis_parser.add_argument("--build-figures", action="store_true")
    regional_analysis_parser.add_argument("--no-figures", action="store_true")

    regional_reviving_offset_parser = sub.add_parser("run-regional-reviving-offset-sensitivity")
    regional_reviving_offset_parser.add_argument("--deployment-run-id", type=str, default="molde4_seed61")
    regional_reviving_offset_parser.add_argument("--run-id")
    regional_reviving_offset_parser.add_argument("--seed", type=int, default=61)
    regional_reviving_offset_parser.add_argument("--offsets", nargs="+", type=float, default=None)
    regional_reviving_offset_parser.add_argument("--chunk-size", type=int, default=2048)
    regional_reviving_offset_parser.add_argument("--num-workers", type=int, default=None)
    regional_reviving_offset_parser.add_argument("--threads-per-worker", type=int, default=1)
    regional_reviving_offset_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    regional_figures_parser = sub.add_parser("build-regional-grid-figures")
    regional_figures_parser.add_argument("--run-id")
    regional_figures_parser.add_argument("--period", choices=REGIONAL_PERIODS[:-1], default="2003_2022")
    regional_figures_parser.add_argument("--climatology-path", type=str, default=None)
    regional_figures_parser.add_argument("--output-dir", type=str, default=None)

    figures_parser = sub.add_parser("build-figures")
    figures_parser.add_argument("--run-id")
    tables_parser = sub.add_parser("build-tables")
    tables_parser.add_argument("--run-id")
    return parser


def ensure_torch_installed() -> None:
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for this project. Install dependencies with `pip install -r requirements.txt`.") from exc


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-data":
        from rice_phenology_hypernet.data.io import prepare_data_assets
        from rice_phenology_hypernet.features import build_modeling_dataset

        prepare_data_assets()
        build_modeling_dataset(force=True)
        return 0

    if args.command == "invert-thresholds":
        from rice_phenology_hypernet.data.io import prepare_data_assets
        from rice_phenology_hypernet.features import compute_threshold_samples

        prepare_data_assets()
        compute_threshold_samples(force=True)
        return 0

    if args.command == "run-dvr-experiment":
        ensure_torch_installed()
        from rice_phenology_hypernet.data.io import prepare_data_assets
        from rice_phenology_hypernet.experiments import run_dvr_experiment
        from rice_phenology_hypernet.features import build_modeling_dataset
        from rice_phenology_hypernet.runtime import initialize_run

        run_paths = initialize_run(run_id=args.run_id)
        prepare_data_assets()
        build_modeling_dataset(force=False)
        run_dvr_experiment(args.task, args.model, force=False, run_id=run_paths.run_id, seed=args.seed)
        return 0

    if args.command == "run-all-dvr-experiments":
        ensure_torch_installed()
        from rice_phenology_hypernet.data.io import prepare_data_assets
        from rice_phenology_hypernet.experiments import run_all_dvr_experiments
        from rice_phenology_hypernet.features import build_modeling_dataset
        from rice_phenology_hypernet.runtime import initialize_run

        run_paths = initialize_run(run_id=args.run_id)
        prepare_data_assets()
        build_modeling_dataset(force=True)
        run_all_dvr_experiments(
            force=False,
            run_id=run_paths.run_id,
            seeds=args.seeds,
            num_workers=args.num_workers,
            threads_per_worker=args.threads_per_worker,
            device=args.device,
            gpu_workers=args.gpu_workers,
        )
        return 0

    if args.command == "train-dvr-deployment-models":
        ensure_torch_installed()
        from rice_phenology_hypernet.data.io import prepare_data_assets
        from rice_phenology_hypernet.experiments import train_dvr_deployment_models
        from rice_phenology_hypernet.features import build_modeling_dataset

        prepare_data_assets()
        build_modeling_dataset(force=False)
        train_dvr_deployment_models(run_id=args.run_id, seed=args.seed)
        return 0

    if args.command == "run-dvr-diagnostic":
        from rice_phenology_hypernet.data.io import prepare_data_assets
        from rice_phenology_hypernet.experiments import run_dvr_diagnostic
        from rice_phenology_hypernet.features import build_modeling_dataset

        prepare_data_assets()
        build_modeling_dataset(force=False)
        run_dvr_diagnostic(args.task, run_id=args.run_id)
        return 0

    if args.command == "analyze-modifier-interpretability":
        ensure_torch_installed()
        from rice_phenology_hypernet.data.io import prepare_data_assets
        from rice_phenology_hypernet.experiments import analyze_modifier_interpretability
        from rice_phenology_hypernet.features import build_modeling_dataset

        prepare_data_assets()
        build_modeling_dataset(force=False)
        analyze_modifier_interpretability(
            deployment_run_id=args.deployment_run_id,
            run_id=args.run_id,
            seed=args.seed,
            stage=args.stage,
        )
        return 0

    if args.command == "prepare-regional-grid-inputs":
        from rice_phenology_hypernet.experiments.regional_grid_projection import prepare_regional_grid_inputs

        result = prepare_regional_grid_inputs(
            period=args.period,
            remote_sensing_path=Path(args.remote_sensing_path) if args.remote_sensing_path else None,
            weather_summary_path=Path(args.weather_summary_path) if args.weather_summary_path else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print("Regional grid inputs prepared:")
        for item in getattr(result, "results", (result,)):
            print(f"  [{item.period}] Valid points: {item.valid_points_path}")
            print(f"  [{item.period}] Point-year inputs: {item.point_year_inputs_path}")
            print(f"  [{item.period}] Excluded points: {item.excluded_points_path}")
        return 0

    if args.command == "run-regional-grid-projection":
        ensure_torch_installed()
        from rice_phenology_hypernet.experiments.regional_grid_projection import run_regional_grid_projection

        result = run_regional_grid_projection(
            deployment_run_id=args.deployment_run_id,
            run_id=args.run_id,
            seed=args.seed,
            period=args.period,
            input_path=Path(args.input_path) if args.input_path else None,
            weather_dir=Path(args.weather_dir) if args.weather_dir else None,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            threads_per_worker=args.threads_per_worker,
            device=args.device,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print("Regional grid projection completed:")
        for item in getattr(result, "results", (result,)):
            print(f"  [{item.period}] Yearly predictions: {item.yearly_predictions_path}")
            print(f"  [{item.period}] Metadata: {item.metadata_path}")
        return 0

    if args.command == "analyze-regional-grid-projection":
        from rice_phenology_hypernet.experiments.regional_grid_analysis import analyze_regional_grid_projection

        result = analyze_regional_grid_projection(
            run_id=args.run_id,
            period=args.period,
            input_dir=Path(args.input_dir) if args.input_dir else None,
            yearly_predictions_path=Path(args.yearly_predictions_path) if args.yearly_predictions_path else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            figures_dir=Path(args.figures_dir) if args.figures_dir else None,
            build_figures=not args.no_figures,
        )
        print("Regional grid projection analysis completed:")
        for item in getattr(result, "results", (result,)):
            print(f"  [{item.period}] Climatology predictions: {item.climatology_predictions_path}")
            print(f"  [{item.period}] Metrics: {item.metrics_path}")
            for figure_path in item.figure_paths:
                print(f"  [{item.period}] Figure: {figure_path}")
            print(f"  [{item.period}] Metadata: {item.metadata_path}")
        if hasattr(result, "period_metrics_path"):
            print(f"  Period metrics: {result.period_metrics_path}")
        return 0

    if args.command == "run-regional-reviving-offset-sensitivity":
        ensure_torch_installed()
        from rice_phenology_hypernet.experiments import run_regional_reviving_offset_sensitivity

        result = run_regional_reviving_offset_sensitivity(
            deployment_run_id=args.deployment_run_id,
            run_id=args.run_id,
            seed=args.seed,
            offsets=args.offsets,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            threads_per_worker=args.threads_per_worker,
            device=args.device,
        )
        print("Regional reviving offset sensitivity completed:")
        print(f"  Run ID: {result.run_id}")
        print(f"  Summary metrics: {result.summary_metrics_path}")
        for item in result.results:
            print(f"  [offset {item.reviving_offset_days:g}] Metrics: {item.metrics_path}")
            for figure_path in item.figure_paths:
                print(f"  [offset {item.reviving_offset_days:g}] Figure: {figure_path}")
        print(f"  Metadata: {result.metadata_path}")
        return 0

    if args.command == "build-regional-grid-figures":
        from rice_phenology_hypernet.experiments.regional_grid_analysis import build_regional_grid_figures

        result = build_regional_grid_figures(
            run_id=args.run_id,
            period=args.period,
            climatology_path=Path(args.climatology_path) if args.climatology_path else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print("Regional grid figures completed:")
        print(f"  Phenology comparison: {result.map_path}")
        if hasattr(result, "residual_diagnostic_path"):
            print(f"  Residual diagnostics: {result.residual_diagnostic_path}")
        if hasattr(result, "scatter_path"):
            print(f"  Paired scatter diagnostics: {result.scatter_path}")
        return 0

    if args.command == "build-figures":
        from rice_phenology_hypernet.figures import build_figures

        build_figures(run_id=args.run_id)
        return 0

    if args.command == "build-tables":
        from rice_phenology_hypernet.tables import build_tables

        build_tables(run_id=args.run_id)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
