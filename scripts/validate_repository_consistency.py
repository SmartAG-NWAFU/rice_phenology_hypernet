#!/usr/bin/env python3
"""Validate the retained public repository against its documented contract."""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "rice_phenology_hypernet"
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml", ".txt"}
PAPER_MODELS = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")
EXPLORATORY_MODEL_TOKENS = (
    "m1_dvr_v4",
    "m1_v3_dvr",
    "m1_dvr_con_rad",
    "m3_direct",
    "m3_seq",
)
README_BANNED = (
    "public supplementary code",
    "## install",
    "pip install",
    "pytest",
    "rice_phenology_hypernet.cli",
    "configs/",
    "tests/",
    "modifier interpretability",
    "reviving-offset sensitivity",
    "figure and table builders",
)


def iter_text_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        yield path


def module_inventory() -> set[str]:
    modules: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(SRC).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.add(".".join(parts))
    return modules


def imported_module(node: ast.ImportFrom, path: Path) -> str | None:
    if node.level == 0:
        return node.module
    relative = path.relative_to(SRC).with_suffix("")
    parts = list(relative.parts)
    parts.pop()
    climb = node.level - 1
    if climb > len(parts):
        return None
    base = parts[: len(parts) - climb]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def check_local_imports(errors: list[str]) -> None:
    inventory = module_inventory()
    for path in PACKAGE.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"AST parse failed: {path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
            continue
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.ImportFrom):
                target = imported_module(node, path)
                if target and target.startswith("rice_phenology_hypernet"):
                    candidates.append(target)
            elif isinstance(node, ast.Import):
                candidates.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("rice_phenology_hypernet")
                )
            for target in candidates:
                if target not in inventory:
                    errors.append(
                        f"Missing local import target: {path.relative_to(ROOT)} -> {target}"
                    )


def check_han(errors: list[str]) -> None:
    han = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
    for path in iter_text_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if han.search(line):
                errors.append(f"Han text: {path.relative_to(ROOT)}:{line_number}")


def check_runner(errors: list[str]) -> None:
    path = PACKAGE / "experiments" / "runner_dvr.py"
    source = path.read_text(encoding="utf-8")
    for token in EXPLORATORY_MODEL_TOKENS:
        if token in source:
            errors.append(f"Exploratory model remains in runner: {token}")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return
    literal_models: tuple[str, ...] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PAPER_MODEL_NAMES"
            for target in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, tuple):
                literal_models = value
    if literal_models is not None and literal_models != PAPER_MODELS:
        errors.append(f"Runner paper model set is {literal_models!r}, expected {PAPER_MODELS!r}")


def check_parameter_defaults(errors: list[str]) -> None:
    managed_fields = {
        "hidden_size",
        "dropout",
        "modifier_cap",
        "event_beta",
        "background_gate_prior",
        "event_loss_weight",
        "terminal_loss_weight",
        "shrink_loss_weight",
        "smooth_loss_weight",
        "mean_anchor_loss_weight",
        "stage_anchor_multipliers",
        "stage_terminal_weights",
        "stage_shrink_multipliers",
        "gate_prior_weight",
        "gate_monotonic_weight",
        "eps",
    }
    for relative in ("models/m1_v2_dvr.py", "models/m1_dvr_con.py", "models/dvr_objective.py"):
        path = PACKAGE / relative
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in managed_fields and node.value is not None:
                    errors.append(f"Config-managed default: {relative}:{node.lineno}:{node.target.id}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = list(node.args.posonlyargs) + list(node.args.args)
                defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
                pairs = list(zip(positional, defaults)) + list(zip(node.args.kwonlyargs, node.args.kw_defaults))
                for argument, default in pairs:
                    if argument.arg in managed_fields and default is not None:
                        errors.append(f"Config-managed function default: {relative}:{node.lineno}:{argument.arg}")
    identifier_fields = {"seed", "deployment_run_id", "run_id"}
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in identifier_fields and node.value is not None:
                    if not isinstance(node.value, ast.Constant) or node.value.value is not None:
                        errors.append(
                            f"Fixed experiment identifier default: {path.relative_to(ROOT)}:{node.lineno}"
                        )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = list(node.args.posonlyargs) + list(node.args.args)
                defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
                pairs = list(zip(positional, defaults)) + list(zip(node.args.kwonlyargs, node.args.kw_defaults))
                for argument, default in pairs:
                    if argument.arg not in identifier_fields or default is None:
                        continue
                    if isinstance(default, ast.Constant) and default.value is None:
                        continue
                    errors.append(
                        f"Fixed experiment identifier default: {path.relative_to(ROOT)}:{node.lineno}"
                    )


def check_exports(errors: list[str]) -> None:
    stale = {
        PACKAGE / "data" / "__init__.py": ("dataset_dvr", "config"),
        PACKAGE / "experiments" / "__init__.py": (
            "modifier_interpretability",
            "regional_reviving_offset_sensitivity",
            "dvr_diagnostic",
            "train_dvr_deployment_models",
            "run_all_dvr_experiments",
        ),
    }
    for path, tokens in stale.items():
        source = path.read_text(encoding="utf-8")
        for token in tokens:
            if token in source:
                errors.append(f"Stale package export: {path.relative_to(ROOT)} -> {token}")


def check_regional_contract(errors: list[str]) -> None:
    projection = (PACKAGE / "experiments" / "regional_grid_projection.py").read_text(encoding="utf-8")
    analysis = (PACKAGE / "experiments" / "regional_grid_analysis.py").read_text(encoding="utf-8")
    for token in ("dataset_dvr", "train_dvr_deployment_models"):
        if token in projection:
            errors.append(f"Regional projection retains omitted infrastructure: {token}")
    for token in ("matplotlib", "build_regional_grid_figures", "FigureBuildResult"):
        if token in analysis:
            errors.append(f"Regional analysis retains figure infrastructure: {token}")


def check_readme(errors: list[str]) -> None:
    source = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = source.lower()
    for phrase in README_BANNED:
        if phrase in lowered:
            errors.append(f"README unsupported or stale claim: {phrase}")


def check_recording_backend(errors: list[str]) -> None:
    sys.path.insert(0, str(SRC))
    try:
        runner = importlib.import_module("rice_phenology_hypernet.experiments.runner_dvr")
    except Exception as exc:  # noqa: BLE001 - validator must accumulate failures
        errors.append(f"Runner import/interface mismatch: {type(exc).__name__}: {exc}")
        return
    required = {"ExperimentSpec", "run_dvr_experiment", "PAPER_MODEL_NAMES"}
    missing = sorted(name for name in required if not hasattr(runner, name))
    if missing:
        errors.append(f"Runner public contract missing: {', '.join(missing)}")
        return
    if tuple(runner.PAPER_MODEL_NAMES) != PAPER_MODELS:
        errors.append("Imported runner paper model set does not match the four-model contract")
        return
    validator = getattr(runner, "validate_recording_backend_contract", None)
    if validator is None:
        errors.append("Runner synthetic recording-backend validator is missing")
        return
    try:
        validator()
    except Exception as exc:  # noqa: BLE001 - validator must accumulate failures
        errors.append(f"Recording-backend contract failed: {type(exc).__name__}: {exc}")


def main() -> int:
    errors: list[str] = []
    check_han(errors)
    check_local_imports(errors)
    check_runner(errors)
    check_parameter_defaults(errors)
    check_exports(errors)
    check_regional_contract(errors)
    check_readme(errors)
    check_recording_backend(errors)
    if errors:
        print("REPOSITORY_CONSISTENCY_CONTRACT_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("REPOSITORY_CONSISTENCY_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
