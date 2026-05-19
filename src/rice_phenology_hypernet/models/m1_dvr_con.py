"""M1-DVR-CON: 阶段衰减背景信息注入的 physics-guided DVR modifier 模型。

核心设计：
- 在日尺度 DVR 修正架构上添加 background_gate 参数控制各阶段背景信息权重
- background_gate 采用 sigmoid 激活，每阶段一个全局标量
- gate 初始化先验: [1.0, 0.75, 0.50, 0.20, 0.10] (tillering → maturity)
- 新增 gate_prior_loss 和 gate_monotonic_loss，约束 gate 向先验靠拢且单调递减

物理解释：
- 前期阶段（分蘖、拔节）：距移栽天数是主要发育驱动 → 背景信息权重高
- 后期阶段（抽穗、成熟）：气温累积主导 → 气象驱动足够，背景信息可能产生年份捷径
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class M1ConDvrConfig:
    """M1-DVR-CON 模型配置。
    
    关键新增参数：
    - background_gate_prior: 各阶段背景信息权重先验（5个阶段）
    """
    input_dim: int = 5
    state_dim: int = 2  # DOY + 距移栽天数
    hidden_size: int = 32
    dropout: float = 0.1
    modifier_cap: float = 2.0
    event_beta: float = 12.0
    background_gate_prior: tuple[float, ...] = (1.0, 0.75, 0.50, 0.20, 0.10)


class M1ConDvrModel(nn.Module):
    """阶段衰减背景信息注入的 DVR modifier 模型。
    
    模型结构：
    1. state_proj: Linear(2, hidden) + Tanh - 将背景信息投影到隐空间
    2. stage_embedding: Embedding(5, hidden) - 阶段类型信息（不受衰减影响）
    3. background_gate: Parameter([5]) - sigmoid 激活，控制背景信息权重
    4. context = stage_embedding + alpha_stage * state_proj
    5. GRU(weather + context) - 时序编码
    6. stage-specific heads - 各阶段修正项输出
    
    注意：stage_embedding 不受衰减影响，因为它提供"阶段类型信息"而非"日历背景"
    """
    
    def __init__(self, config: M1ConDvrConfig | None = None):
        super().__init__()
        self.config = config or M1ConDvrConfig()
        
        # 状态投影（背景信息）
        self.state_proj = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_size),
            nn.Tanh(),
        )
        
        # 阶段 embedding（不衰减）
        self.stage_embedding = nn.Embedding(5, self.config.hidden_size)
        
        # 背景信息 gate 参数：每阶段一个标量
        # 初始化为 sigmoid 的 inverse，使得 sigmoid(gate_logits) ≈ prior
        prior = torch.tensor(self.config.background_gate_prior, dtype=torch.float32)
        # sigmoid inverse: log(p / (1-p))，处理边界情况
        gate_logits_init = torch.log(prior / (1.0 - prior + 1e-6))
        self.background_gate_logits = nn.Parameter(gate_logits_init)
        
        # GRU 编码器
        self.gru = nn.GRU(
            input_size=self.config.input_dim + self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        self.dropout = nn.Dropout(self.config.dropout)
        
        # 阶段特异 heads
        self.stage_heads = nn.ModuleList(
            nn.Linear(self.config.hidden_size, 1) for _ in range(5)
        )
    
    def get_background_gates(self) -> torch.Tensor:
        """获取当前各阶段的背景信息权重 (sigmoid 激活后)。"""
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
        """前向传播，带阶段衰减。
        
        Args:
            weather_seq: [batch, seq_len, input_dim]
            stage_state: [batch, 2] - DOY + 距移栽天数
            stage_index: [batch] - 阶段索引 (0-4)
            base_dvr_seq: [batch, seq_len]
            mask: [batch, seq_len]
        
        Returns:
            dict 包含：
            - log_modifier_seq: [batch, seq_len]
            - modifier_seq: [batch, seq_len]
            - dvr_star_seq: [batch, seq_len]
            - cum_progress_seq: [batch, seq_len]
            - completion_cdf: [batch, seq_len]
            - all_background_gates: [5] - 所有阶段的 gate 值
            - stage_background_gate: [batch] - 当前 batch 各样本的阶段 gate
        """
        # 获取所有阶段的 gate 值
        all_gates = self.get_background_gates()  # [5]
        
        # 获取 batch 中每个样本的阶段 gate
        stage_gates = all_gates[stage_index]  # [batch]
        
        # 投影背景信息
        state_proj_out = self.state_proj(stage_state)  # [batch, hidden]
        
        # 加权背景信息 + stage_embedding（不衰减）
        # context = stage_embedding + gate * state_proj
        context = self.stage_embedding(stage_index) + stage_gates.unsqueeze(-1) * state_proj_out
        
        # 展开为序列
        context_seq = context.unsqueeze(1).expand(-1, weather_seq.shape[1], -1)
        
        # GRU 编码
        encoded, _ = self.gru(torch.cat([weather_seq, context_seq], dim=-1))
        encoded = self.dropout(encoded)
        
        # 阶段特异 heads
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
    stage_index: torch.Tensor | None = None,
    event_loss_weight: float = 1.0,
    terminal_loss_weight: float = 1.0,
    shrink_loss_weight: float = 0.05,
    smooth_loss_weight: float = 0.01,
    mean_anchor_loss_weight: float = 0.0,
    stage_anchor_multipliers: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    stage_terminal_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    stage_shrink_multipliers: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    gate_prior_weight: float = 0.05,
    gate_monotonic_weight: float = 0.05,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算条件 DVR 修正模型的损失，包含 gate 正则项。
    
    新增损失项：
    - gate_prior_loss: 约束 gate 向先验靠拢
    - gate_monotonic_loss: 约束 gate 单调递减 (tillering ≥ jointing ≥ ... ≥ maturity)
    """
    from .dvr_loss import compute_dvr_loss
    
    # 先计算基础 DVR loss
    base_loss, base_stats = compute_dvr_loss(
        outputs,
        true_duration,
        mask,
        stage_index=stage_index,
        event_loss_weight=event_loss_weight,
        terminal_loss_weight=terminal_loss_weight,
        shrink_loss_weight=shrink_loss_weight,
        smooth_loss_weight=smooth_loss_weight,
        mean_anchor_loss_weight=mean_anchor_loss_weight,
        stage_anchor_multipliers=stage_anchor_multipliers,
        stage_terminal_weights=stage_terminal_weights,
        stage_shrink_multipliers=stage_shrink_multipliers,
        eps=eps,
    )
    
    # Gate 正则项
    prior = torch.tensor(model.config.background_gate_prior, dtype=torch.float32, device=true_duration.device)
    current_gates = model.get_background_gates()  # [5]
    
    # gate_prior_loss: MSE between current gates and prior
    gate_prior_loss = torch.mean((current_gates - prior) ** 2)
    
    # gate_monotonic_loss: penalize non-monotonic (later stage should have lower gate)
    # relu(alpha[1:] - alpha[:-1]) ensures alpha[i] >= alpha[i+1]
    gate_monotonic_loss = torch.mean(torch.relu(current_gates[1:] - current_gates[:-1]))
    
    # 总损失
    total_loss = base_loss + gate_prior_weight * gate_prior_loss + gate_monotonic_weight * gate_monotonic_loss
    
    # 合并 stats
    stats = base_stats | {
        "gate_prior_loss": float(gate_prior_loss.item()),
        "gate_monotonic_loss": float(gate_monotonic_loss.item()),
    }
    
    return total_loss, stats


__all__ = ["M1ConDvrConfig", "M1ConDvrModel", "compute_m1_dvr_con_loss"]
