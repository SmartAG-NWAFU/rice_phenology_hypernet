from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class M1V2DvrConfig:
    """Model settings populated from the experiment configuration."""

    hidden_size: int
    dropout: float
    modifier_cap: float
    event_beta: float
    input_dim: int = 5


class M1V2DvrModel(nn.Module):
    def __init__(self, config: M1V2DvrConfig):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.stage_heads = nn.ModuleList(nn.Linear(self.config.hidden_size, 1) for _ in range(5))

    def forward(
        self,
        *,
        weather_seq: torch.Tensor,
        stage_index: torch.Tensor,
        base_dvr_seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded, _ = self.gru(weather_seq)
        encoded = self.dropout(encoded)

        head_logits = torch.stack([head(encoded).squeeze(-1) for head in self.stage_heads], dim=-1)
        gather_index = stage_index.view(-1, 1, 1).expand(-1, weather_seq.shape[1], 1)
        log_modifier = head_logits.gather(2, gather_index).squeeze(-1)
        log_modifier = torch.clamp(log_modifier, min=-self.config.modifier_cap, max=self.config.modifier_cap)
        modifier_seq = torch.exp(log_modifier)
        modifier_seq = torch.where(mask, modifier_seq, torch.ones_like(modifier_seq))
        dvr_star_seq = base_dvr_seq * modifier_seq * mask.float()
        cum_progress_seq = torch.cumsum(dvr_star_seq, dim=1)
        completion_cdf = torch.sigmoid(self.config.event_beta * (cum_progress_seq - 1.0))

        return {
            "log_modifier_seq": log_modifier,
            "modifier_seq": modifier_seq,
            "dvr_star_seq": dvr_star_seq,
            "cum_progress_seq": cum_progress_seq,
            "completion_cdf": completion_cdf,
        }


__all__ = ["M1V2DvrConfig", "M1V2DvrModel"]
