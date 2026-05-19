import json

import torch

import rice_phenology_hypernet.experiments.runner_dvr as runner_dvr
from rice_phenology_hypernet.experiments.runner_dvr import (
    DEPLOYMENT_MODEL_NAMES,
    MaterializedProcessModel,
    load_dvr_deployment_artifact,
    materialize_dvr_deployment_artifact,
)
from rice_phenology_hypernet.models.m1_v2_dvr import M1V2DvrConfig, M1V2DvrModel
from rice_phenology_hypernet.settings import build_project_settings


STAGE_REQUIREMENTS = {
    "tillering": 8.0,
    "jointing": 10.0,
    "booting": 12.0,
    "heading": 14.0,
    "maturity": 16.0,
}


def _write_process_artifact(root, run_id, seed, model_name):
    model_dir = root / "artifacts" / "models" / run_id / f"seed_{seed}" / model_name
    model_dir.mkdir(parents=True)
    (model_dir / "stage_requirements.json").write_text(json.dumps(STAGE_REQUIREMENTS), encoding="utf-8")
    (model_dir / "deployment_metadata.json").write_text(
        json.dumps({"artifact_type": "process", "model_name": model_name}),
        encoding="utf-8",
    )


def _write_learned_artifact(root, run_id, seed):
    model = M1V2DvrModel(M1V2DvrConfig(hidden_size=4, dropout=0.0))
    model_dir = root / "artifacts" / "models" / run_id / f"seed_{seed}" / "m1_v2_dvr"
    model_dir.mkdir(parents=True)
    (model_dir / "stage_requirements.json").write_text(json.dumps(STAGE_REQUIREMENTS), encoding="utf-8")
    (model_dir / "deployment_metadata.json").write_text(
        json.dumps({"artifact_type": "learned", "model_name": "m1_v2_dvr"}),
        encoding="utf-8",
    )
    torch.save(
        {
            "model": "m1_v2_dvr",
            "model_config": model.config.__dict__,
            "model_state_dict": model.state_dict(),
            "stage_requirements": STAGE_REQUIREMENTS,
        },
        model_dir / "model.pt",
    )


def test_public_deployment_model_list_is_limited_to_paper_models():
    assert DEPLOYMENT_MODEL_NAMES == ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")


def test_process_deployment_artifacts_load_and_materialize(monkeypatch, tmp_path):
    settings = build_project_settings(tmp_path)
    monkeypatch.setattr(runner_dvr, "SETTINGS", settings)
    _write_process_artifact(tmp_path, "public_run", 61, "m0_t")

    artifact = load_dvr_deployment_artifact("public_run", "m0_t", seed=61)
    materialized = materialize_dvr_deployment_artifact(artifact)

    assert artifact.artifact_type == "process"
    assert artifact.stage_requirements == STAGE_REQUIREMENTS
    assert isinstance(materialized, MaterializedProcessModel)
    assert materialized.model_name == "m0_t"


def test_learned_deployment_artifact_loads_and_returns_eval_module(monkeypatch, tmp_path):
    settings = build_project_settings(tmp_path)
    monkeypatch.setattr(runner_dvr, "SETTINGS", settings)
    _write_learned_artifact(tmp_path, "public_run", 61)

    artifact = load_dvr_deployment_artifact("public_run", "m1_v2_dvr", seed=61)
    model = materialize_dvr_deployment_artifact(artifact)

    assert artifact.artifact_type == "learned"
    assert isinstance(model, torch.nn.Module)
    assert not model.training
