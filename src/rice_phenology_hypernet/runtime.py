from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rice_phenology_hypernet.settings import ProjectSettings, SETTINGS


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    eval_dir: Path
    figures_dir: Path
    tables_dir: Path
    config_snapshot_dir: Path
    manifest_path: Path


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    return current.strftime("%Y%m%d_%H%M%S")


def get_run_paths(run_id: str, settings: ProjectSettings = SETTINGS) -> RunPaths:
    return RunPaths(
        run_id=run_id,
        eval_dir=settings.eval_dir / run_id,
        figures_dir=settings.figures_dir / run_id,
        tables_dir=settings.tables_dir / run_id,
        config_snapshot_dir=settings.eval_dir / run_id / "config_snapshot",
        manifest_path=settings.eval_dir / run_id / "run_manifest.json",
    )


def _latest_path(settings: ProjectSettings) -> Path:
    return settings.eval_dir / "latest.json"


def _write_latest(run_id: str, settings: ProjectSettings = SETTINGS) -> None:
    payload = {"run_id": run_id, "updated_at": datetime.now().astimezone().isoformat()}
    _latest_path(settings).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_latest_run_id(settings: ProjectSettings = SETTINGS) -> str:
    path = _latest_path(settings)
    if not path.exists():
        raise FileNotFoundError("No latest run found in artifacts/eval. Run an experiment first or pass --run-id.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("artifacts/eval/latest.json is missing a valid run_id")
    return run_id


def initialize_run(run_id: str | None = None, settings: ProjectSettings = SETTINGS, update_latest: bool = True) -> RunPaths:
    effective_run_id = run_id or generate_run_id()
    paths = get_run_paths(effective_run_id, settings=settings)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.config_snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not paths.manifest_path.exists():
        for config_path in settings.configs_dir.glob("*.yaml"):
            shutil.copy2(config_path, paths.config_snapshot_dir / config_path.name)
        payload = {
            "run_id": effective_run_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "config_snapshot_dir": str(paths.config_snapshot_dir.relative_to(settings.root)),
            "experiments": [],
        }
        paths.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if update_latest:
        _write_latest(effective_run_id, settings=settings)
    return paths


def require_run(run_id: str | None = None, settings: ProjectSettings = SETTINGS) -> RunPaths:
    effective_run_id = run_id or get_latest_run_id(settings=settings)
    paths = get_run_paths(effective_run_id, settings=settings)
    if not paths.eval_dir.exists():
        raise FileNotFoundError(f"Missing eval directory for run_id '{effective_run_id}': {paths.eval_dir}")
    return paths


def _load_manifest(paths: RunPaths) -> dict:
    if not paths.manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {paths.manifest_path}")
    return json.loads(paths.manifest_path.read_text(encoding="utf-8"))


def _resolve_manifest_output_dir(value: str, settings: ProjectSettings) -> Path:
    path = Path(value)
    return path if path.is_absolute() else settings.root / path


def resolve_seed_eval_dirs(
    run_paths: RunPaths | None = None,
    *,
    run_id: str | None = None,
    settings: ProjectSettings = SETTINGS,
) -> tuple[Path, ...]:
    paths = run_paths or require_run(run_id=run_id, settings=settings)
    manifest_dirs: list[Path] = []
    seen: set[Path] = set()

    if paths.manifest_path.exists():
        payload = _load_manifest(paths)
        for record in payload.get("experiments", []):
            candidate: Path | None = None
            output_dir = record.get("output_dir")
            if isinstance(output_dir, str) and output_dir:
                resolved = _resolve_manifest_output_dir(output_dir, settings).resolve()
                if resolved.is_dir():
                    candidate = resolved
            if candidate is None and "seed" in record:
                seed_dir = (paths.eval_dir / f"seed_{int(record['seed'])}").resolve()
                if seed_dir.is_dir():
                    candidate = seed_dir
            if candidate is not None and candidate not in seen:
                seen.add(candidate)
                manifest_dirs.append(candidate)
    if manifest_dirs:
        return tuple(manifest_dirs)

    seed_dirs = tuple(sorted(path.resolve() for path in paths.eval_dir.glob("seed_*") if path.is_dir()))
    if seed_dirs:
        return seed_dirs

    return (paths.eval_dir.resolve(),)


def _write_manifest(paths: RunPaths, payload: dict) -> None:
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    paths.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_run_metadata(run_id: str, settings: ProjectSettings = SETTINGS, **updates: object) -> None:
    paths = initialize_run(run_id=run_id, settings=settings, update_latest=False)
    payload = _load_manifest(paths)
    payload.update(updates)
    _write_manifest(paths, payload)


def _manifest_output_dir(value: Path | str, settings: ProjectSettings) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(settings.root))
    except ValueError:
        return str(path)


def register_experiment(
    run_id: str,
    task: str,
    model: str,
    *,
    seed: int | None = None,
    output_dir: Path | str | None = None,
    settings: ProjectSettings = SETTINGS,
) -> None:
    paths = initialize_run(run_id=run_id, settings=settings, update_latest=False)
    payload = _load_manifest(paths)
    experiments = payload.setdefault("experiments", [])
    record = {"task": task, "model": model}
    if seed is not None:
        record["seed"] = int(seed)
    if output_dir is not None:
        record["output_dir"] = _manifest_output_dir(output_dir, settings)
    if record not in experiments:
        experiments.append(record)
    _write_manifest(paths, payload)
