from rice_phenology_hypernet.config import (
    PUBLIC_MODEL_NAMES,
    get_default_feature_set_for_model,
    get_generated_feature_names,
    get_project_config,
)


def test_public_config_limits_tasks_models_and_batch_seed():
    config = get_project_config()

    assert config.experiment.sample.n_splits > 1
    assert config.experiment.site.n_splits > 1
    assert len(config.experiment.year.folds) >= 1
    assert config.experiment.dvr_batch.seeds == (61,)
    assert PUBLIC_MODEL_NAMES == ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")
    assert config.paper.comparison_models == PUBLIC_MODEL_NAMES
    assert config.paper.latest_model.model == "m1_dvr_con"


def test_generated_features_and_model_feature_sets_are_consistent():
    config = get_project_config()
    generated = set(get_generated_feature_names())

    assert generated
    assert set(config.features.generated_feature_names) == generated
    for model_name in PUBLIC_MODEL_NAMES:
        feature_set = set(get_default_feature_set_for_model(model_name))
        assert feature_set <= generated
        assert feature_set


def test_learned_dvr_configs_have_required_stage_weights():
    config = get_project_config()

    for model_config in (config.experiment.m1_v2_dvr, config.experiment.m1_dvr_con):
        assert model_config.epochs > 0
        assert model_config.batch_size > 0
        assert model_config.max_sequence_length > 0
        assert len(model_config.stage_anchor_multipliers) == 5
        assert len(model_config.stage_terminal_weights) == 5
        assert len(model_config.stage_shrink_multipliers) == 5

    assert len(config.experiment.m1_dvr_con.background_gate_prior) == 5
    assert config.experiment.m1_dvr_con.gate_prior_weight >= 0
    assert config.experiment.m1_dvr_con.gate_monotonic_weight >= 0
