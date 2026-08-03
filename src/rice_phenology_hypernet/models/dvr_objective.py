"""Configuration-injected objective for daily DVR correction models."""

from __future__ import annotations

from typing import Protocol

import torch


class DvrLossConfig(Protocol):
    """Configuration-owned settings required by the DVR objective."""

    event_loss_weight: float
    terminal_loss_weight: float
    shrink_loss_weight: float
    smooth_loss_weight: float
    mean_anchor_loss_weight: float
    stage_anchor_multipliers: tuple[float, ...]
    stage_terminal_weights: tuple[float, ...]
    stage_shrink_multipliers: tuple[float, ...]
    eps: float


def first_crossing_day(
    cum_progress_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the first one-based day reaching unit progress for each sample."""

    crossed = (cum_progress_seq >= 1.0) & mask
    any_crossed = crossed.any(dim=1)
    first_cross = torch.argmax(crossed.int(), dim=1) + 1
    fallback = mask.sum(dim=1)
    return torch.where(any_crossed, first_cross, fallback)


def compute_dvr_loss(
    outputs: dict[str, torch.Tensor],
    true_duration: torch.Tensor,
    mask: torch.Tensor,
    *,
    config: DvrLossConfig,
    stage_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the event, terminal, shrinkage, smoothness, and anchor losses."""

    batch_index = torch.arange(true_duration.shape[0], device=true_duration.device)
    duration_index = torch.clamp(true_duration.long() - 1, min=0)

    completion_cdf = outputs["completion_cdf"]
    cum_progress = outputs["cum_progress_seq"]
    log_modifier = outputs["log_modifier_seq"]

    cdf_now = completion_cdf[batch_index, duration_index]
    prev_index = torch.clamp(duration_index - 1, min=0)
    cdf_prev = completion_cdf[batch_index, prev_index]
    cdf_prev = torch.where(
        duration_index > 0,
        cdf_prev,
        torch.zeros_like(cdf_prev),
    )
    event_prob = torch.clamp(cdf_now - cdf_prev, min=config.eps)
    event_loss = -torch.log(event_prob).mean()

    terminal_progress = cum_progress[batch_index, duration_index]
    terminal_residual = (terminal_progress - 1.0) ** 2

    sample_stage_index = (
        stage_index.long()
        if stage_index is not None
        else torch.zeros_like(true_duration, dtype=torch.long)
    )
    anchor_weights = torch.tensor(
        config.stage_anchor_multipliers,
        dtype=torch.float32,
        device=true_duration.device,
    )
    terminal_weights = torch.tensor(
        config.stage_terminal_weights,
        dtype=torch.float32,
        device=true_duration.device,
    )
    shrink_weights = torch.tensor(
        config.stage_shrink_multipliers,
        dtype=torch.float32,
        device=true_duration.device,
    )

    anchor_weight = anchor_weights[sample_stage_index]
    terminal_weight = terminal_weights[sample_stage_index]
    shrink_weight = shrink_weights[sample_stage_index].unsqueeze(1)

    terminal_loss = torch.mean(terminal_residual)
    weighted_terminal_loss = torch.mean(terminal_residual * terminal_weight)

    valid_mask = mask.float()
    shrink_denom = torch.clamp(valid_mask.sum(), min=1.0)
    shrink_loss = (
        torch.sum((log_modifier**2) * valid_mask * shrink_weight) / shrink_denom
    )

    mean_log_modifier = torch.sum(log_modifier * valid_mask, dim=1) / torch.clamp(
        valid_mask.sum(dim=1),
        min=1.0,
    )
    mean_anchor_loss = torch.mean((mean_log_modifier**2) * anchor_weight)

    smooth_mask = mask[:, 1:] & mask[:, :-1]
    smooth_denom = torch.clamp(smooth_mask.float().sum(), min=1.0)
    smooth_loss = torch.sum(
        ((log_modifier[:, 1:] - log_modifier[:, :-1]) ** 2)
        * smooth_mask.float()
    ) / smooth_denom

    total_loss = (
        config.event_loss_weight * event_loss
        + config.terminal_loss_weight * weighted_terminal_loss
        + config.shrink_loss_weight * shrink_loss
        + config.smooth_loss_weight * smooth_loss
        + config.mean_anchor_loss_weight * mean_anchor_loss
    )

    pred_duration = first_crossing_day(cum_progress, mask)
    mae_duration = torch.mean(
        torch.abs(pred_duration.float() - true_duration.float())
    )
    stats = {
        "event_loss": float(event_loss.item()),
        "terminal_loss": float(terminal_loss.item()),
        "weighted_terminal_loss": float(weighted_terminal_loss.item()),
        "shrink_loss": float(shrink_loss.item()),
        "smooth_loss": float(smooth_loss.item()),
        "mean_anchor_loss": float(mean_anchor_loss.item()),
        "mae_duration": float(mae_duration.item()),
    }
    return total_loss, stats


__all__ = ["DvrLossConfig", "compute_dvr_loss", "first_crossing_day"]
