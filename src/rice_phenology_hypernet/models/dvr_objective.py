"""Configuration-injected objective for daily DVR correction models."""

from __future__ import annotations

from typing import Protocol

import torch

from rice_phenology_hypernet.experiments.dvr_core import MAX_TRANSITION_DAYS


class DvrLossConfig(Protocol):
    """Configuration-owned settings required by the DVR objective."""

    event_loss_weight: float
    terminal_loss_weight: float
    shrink_loss_weight: float
    smooth_loss_weight: float
    eps: float


def first_crossing_day(
    cum_progress_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the first one-based day reaching unit progress for each sample."""

    within_window = (
        torch.arange(mask.shape[1], device=mask.device) < MAX_TRANSITION_DAYS
    ).unsqueeze(0)
    rollout_mask = mask & within_window
    crossed = (cum_progress_seq >= 1.0) & rollout_mask
    any_crossed = crossed.any(dim=1)
    first_cross = torch.argmax(crossed.int(), dim=1) + 1
    day_index = torch.arange(
        1,
        mask.shape[1] + 1,
        device=mask.device,
    ).unsqueeze(0)
    fallback = torch.where(rollout_mask, day_index, 0).max(dim=1).values
    return torch.where(any_crossed, first_cross, fallback)


def compute_dvr_loss(
    outputs: dict[str, torch.Tensor],
    true_duration: torch.Tensor,
    mask: torch.Tensor,
    *,
    config: DvrLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the four DRC loss terms defined in the manuscript."""

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

    terminal_loss = torch.mean(terminal_residual)

    valid_mask = mask.float()
    shrink_denom = torch.clamp(valid_mask.sum(), min=1.0)
    shrink_loss = torch.sum((log_modifier**2) * valid_mask) / shrink_denom

    smooth_mask = mask[:, 1:] & mask[:, :-1]
    smooth_denom = torch.clamp(smooth_mask.float().sum(), min=1.0)
    smooth_loss = torch.sum(
        ((log_modifier[:, 1:] - log_modifier[:, :-1]) ** 2)
        * smooth_mask.float()
    ) / smooth_denom

    total_loss = (
        config.event_loss_weight * event_loss
        + config.terminal_loss_weight * terminal_loss
        + config.shrink_loss_weight * shrink_loss
        + config.smooth_loss_weight * smooth_loss
    )

    pred_duration = first_crossing_day(cum_progress, mask)
    mae_duration = torch.mean(
        torch.abs(pred_duration.float() - true_duration.float())
    )
    stats = {
        "event_loss": float(event_loss.item()),
        "terminal_loss": float(terminal_loss.item()),
        "shrink_loss": float(shrink_loss.item()),
        "smooth_loss": float(smooth_loss.item()),
        "mae_duration": float(mae_duration.item()),
    }
    return total_loss, stats


__all__ = ["DvrLossConfig", "compute_dvr_loss", "first_crossing_day"]
