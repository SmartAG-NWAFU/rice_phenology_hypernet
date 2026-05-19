import torch

from rice_phenology_hypernet.models.dvr_loss import compute_dvr_loss, first_crossing_day
from rice_phenology_hypernet.models.m1_dvr_con import M1ConDvrConfig, M1ConDvrModel, compute_m1_dvr_con_loss
from rice_phenology_hypernet.models.m1_v2_dvr import M1V2DvrConfig, M1V2DvrModel


def _common_inputs(batch_size: int = 3, seq_len: int = 8):
    weather_seq = torch.randn(batch_size, seq_len, 5)
    stage_index = torch.tensor([0, 2, 4], dtype=torch.long)[:batch_size]
    base_dvr_seq = torch.full((batch_size, seq_len), 0.16)
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    true_duration = torch.tensor([5, 6, 7], dtype=torch.long)[:batch_size]
    return weather_seq, stage_index, base_dvr_seq, mask, true_duration


def test_first_crossing_day_falls_back_to_valid_sequence_length():
    cum_progress = torch.tensor([[0.2, 0.8, 1.1, 1.4], [0.1, 0.2, 0.3, 0.0]])
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

    assert first_crossing_day(cum_progress, mask).tolist() == [3, 3]


def test_public_dvr_model_forward_shapes_and_loss():
    weather_seq, stage_index, base_dvr_seq, mask, true_duration = _common_inputs()
    model = M1V2DvrModel(M1V2DvrConfig(hidden_size=6, dropout=0.0))

    outputs = model(
        weather_seq=weather_seq,
        stage_index=stage_index,
        base_dvr_seq=base_dvr_seq,
        mask=mask,
    )
    loss, stats = compute_dvr_loss(outputs, true_duration, mask, stage_index=stage_index)

    assert outputs["cum_progress_seq"].shape == base_dvr_seq.shape
    assert outputs["modifier_seq"].shape == base_dvr_seq.shape
    assert loss.ndim == 0
    assert stats["mae_duration"] >= 0.0


def test_constrained_dvr_model_uses_stage_state_and_gate_regularization():
    weather_seq, stage_index, base_dvr_seq, mask, true_duration = _common_inputs()
    model = M1ConDvrModel(M1ConDvrConfig(hidden_size=6, dropout=0.0))
    stage_state = torch.tensor([[100.0, 20.0], [140.0, 60.0], [180.0, 100.0]], dtype=torch.float32)

    outputs = model(
        weather_seq=weather_seq,
        stage_state=stage_state,
        stage_index=stage_index,
        base_dvr_seq=base_dvr_seq,
        mask=mask,
    )
    loss, stats = compute_m1_dvr_con_loss(outputs, true_duration, mask, model=model, stage_index=stage_index)

    assert outputs["all_background_gates"].shape == (5,)
    assert outputs["stage_background_gate"].shape == (3,)
    assert torch.all((outputs["all_background_gates"] >= 0.0) & (outputs["all_background_gates"] <= 1.0))
    assert loss.ndim == 0
    assert "gate_prior_loss" in stats
    assert "gate_monotonic_loss" in stats
