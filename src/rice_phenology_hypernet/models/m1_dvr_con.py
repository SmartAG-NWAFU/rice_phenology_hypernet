"""M1-DVR-CON: a physics-guided DVR modifier model with stage-decaying background-information injection.

Core design:
- Adds a background_gate parameter to the daily DVR correction architecture to control the
  background-information weight for each stage.
- Applies sigmoid activation to background_gate, with one global scalar per stage.
- Initializes gates from a stage-dependent prior supplied by the experiment configuration.
- Adds gate_prior_loss and gate_monotonic_loss to constrain gates toward the prior and enforce
  a monotonic decrease.

Physical interpretation:
- Early stages (tillering and jointing): days since transplanting are the primary development
  driver → high background-information weights.
- Later stages (heading and maturity): accumulated temperature dominates → weather drivers are
  sufficient, and background information may introduce a year-related shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .dvr_objective import DvrLossConfig, compute_dvr_loss


class ConstrainedDvrLossConfig(DvrLossConfig, Protocol):
    """Configuration-owned settings required by the constrained DVR loss."""

    gate_prior_weight: float
    gate_monotonic_weight: float


@dataclass(frozen=True)
class M1ConDvrConfig:
    """Model settings populated from the experiment configuration."""

    hidden_size: int
    dropout: float
    modifier_cap: float
    event_beta: float
    background_gate_prior: tuple[float, ...]
    input_dim: int = 5
    state_dim: int = 2  # DOY + days since transplanting


class M1ConDvrModel(nn.Module):
    """DVR modifier model with stage-decaying background-information injection.
    
    Model architecture:
    1. state_proj: Linear(2, hidden) + Tanh - projects background information into hidden space
    2. stage_embedding: Embedding(5, hidden) - stage-type information (not subject to decay)
    3. background_gate: Parameter([5]) - sigmoid activation controlling background-information weights
    4. context = stage_embedding + alpha_stage * state_proj
    5. GRU(weather + context) - temporal encoding
    6. stage-specific heads - stage-specific correction outputs
    
    Note: stage_embedding is not subject to decay because it provides stage-type information
    rather than calendar context.
    """
    
    def __init__(self, config: M1ConDvrConfig):
        super().__init__()
        self.config = config
        
        # State projection for background information.
        self.state_proj = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_size),
            nn.Tanh(),
        )
        
        # Stage embedding without decay.
        self.stage_embedding = nn.Embedding(5, self.config.hidden_size)
        
        # Background-information gate parameter: one scalar per stage.
        # Initialize with the inverse sigmoid so that sigmoid(gate_logits) ≈ prior.
        prior = torch.tensor(self.config.background_gate_prior, dtype=torch.float32)
        # Inverse sigmoid: log(p / (1-p)), with boundary handling.
        gate_logits_init = torch.log(
            prior / (1.0 - prior + torch.finfo(prior.dtype).eps)
        )
        self.background_gate_logits = nn.Parameter(gate_logits_init)
        
        # GRU encoder.
        self.gru = nn.GRU(
            input_size=self.config.input_dim + self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Stage-specific heads.
        self.stage_heads = nn.ModuleList(
            nn.Linear(self.config.hidden_size, 1) for _ in range(5)
        )
    
    def get_background_gates(self) -> torch.Tensor:
        """Return the current background-information weight for each stage after sigmoid activation."""
        return torch.sigmoid(self.background_gate_logits)
    
    def forward(
        self,
        *,
        weather_seq: torch.Tensor,
        stage_state: torch.Tensor,
        stage_index: torch.Tensor,
        base_dvr_seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the forward pass with stage decay.
        
        Args:
            weather_seq: [batch, seq_len, input_dim]
            stage_state: [batch, 2] - DOY + days since transplanting
            stage_index: [batch] - stage index (0-4)
            base_dvr_seq: [batch, seq_len]
            mask: [batch, seq_len]
        
        Returns:
            Dictionary containing:
            - log_modifier_seq: [batch, seq_len]
            - modifier_seq: [batch, seq_len]
            - dvr_star_seq: [batch, seq_len]
            - cum_progress_seq: [batch, seq_len]
            - completion_cdf: [batch, seq_len]
            - all_background_gates: [5] - gate values for all stages
            - stage_background_gate: [batch] - stage gate for each sample in the current batch
        """
        # Get the gate values for all stages.
        all_gates = self.get_background_gates()  # [5]
        
        # Get the stage gate for each sample in the batch.
        stage_gates = all_gates[stage_index]  # [batch]
        
        # Project the background information.
        state_proj_out = self.state_proj(stage_state)  # [batch, hidden]
        
        # Combine weighted background information with the non-decaying stage_embedding.
        # context = stage_embedding + gate * state_proj
        context = self.stage_embedding(stage_index) + stage_gates.unsqueeze(-1) * state_proj_out
        
        # Expand across the sequence.
        context_seq = context.unsqueeze(1).expand(-1, weather_seq.shape[1], -1)
        
        # GRU encoding.
        encoded, _ = self.gru(torch.cat([weather_seq, context_seq], dim=-1))
        encoded = self.dropout(encoded)
        
        # Stage-specific heads.
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
            "all_background_gates": all_gates,
            "stage_background_gate": stage_gates,
        }


def compute_m1_dvr_con_loss(
    outputs: dict[str, torch.Tensor],
    true_duration: torch.Tensor,
    mask: torch.Tensor,
    *,
    model: M1ConDvrModel,
    config: ConstrainedDvrLossConfig,
    stage_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the conditional DVR correction loss, including gate regularization.
    
    Additional loss terms:
    - gate_prior_loss: constrains gates toward the prior
    - gate_monotonic_loss: enforces monotonically decreasing gates
      (tillering ≥ jointing ≥ ... ≥ maturity)
    """
    # First compute the base DVR loss.
    base_loss, base_stats = compute_dvr_loss(
        outputs,
        true_duration,
        mask,
        config=config,
        stage_index=stage_index,
    )
    
    # Gate regularization terms.
    prior = torch.tensor(model.config.background_gate_prior, dtype=torch.float32, device=true_duration.device)
    current_gates = model.get_background_gates()  # [5]
    
    # gate_prior_loss: MSE between current gates and prior
    gate_prior_loss = torch.mean((current_gates - prior) ** 2)
    
    # gate_monotonic_loss: penalize non-monotonic (later stage should have lower gate)
    # relu(alpha[1:] - alpha[:-1]) ensures alpha[i] >= alpha[i+1]
    gate_monotonic_loss = torch.mean(torch.relu(current_gates[1:] - current_gates[:-1]))
    
    # Total loss.
    total_loss = (
        base_loss
        + config.gate_prior_weight * gate_prior_loss
        + config.gate_monotonic_weight * gate_monotonic_loss
    )
    
    # Merge statistics.
    stats = base_stats | {
        "gate_prior_loss": float(gate_prior_loss.item()),
        "gate_monotonic_loss": float(gate_monotonic_loss.item()),
    }
    
    return total_loss, stats


__all__ = ["M1ConDvrConfig", "M1ConDvrModel", "compute_m1_dvr_con_loss"]
