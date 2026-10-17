"""Land-conditioned low-rank modulation of the LSTM forget gate."""
from __future__ import annotations

from typing import Literal
import torch
from torch import Tensor, nn
import torch.nn.functional as F

AdapterMode = Literal["none", "conditional", "shared", "static_bias"]


class LandMemoryLSTM(nn.Module):
    """Concatenation LSTM with a low-rank additive forget-logit branch."""

    def __init__(
        self,
        dynamic_size: int,
        static_size: int,
        land_size: int,
        hidden_size: int = 64,
        rank: int = 4,
        dropout: float = 0.2,
        adapter_mode: AdapterMode = "conditional",
    ) -> None:
        super().__init__()
        if min(dynamic_size, hidden_size, rank) <= 0 or min(static_size, land_size) < 0:
            raise ValueError("Feature sizes must be valid; hidden_size and rank must be positive.")
        if adapter_mode not in {"none", "conditional", "shared", "static_bias"}:
            raise ValueError(f"Unknown adapter_mode: {adapter_mode}")
        if adapter_mode in {"conditional", "static_bias"} and land_size == 0:
            raise ValueError("This adapter mode requires land attributes.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.dynamic_size = dynamic_size
        self.static_size = static_size
        self.land_size = land_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.adapter_mode = adapter_mode
        self.core = nn.LSTM(dynamic_size + static_size, hidden_size, batch_first=True)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

        # PyTorch gate order: input, forget, candidate, output (i, f, g, o).
        # The two forget biases sum to one.
        with torch.no_grad():
            self.core.bias_ih_l0.zero_()
            self.core.bias_hh_l0.zero_()
            self.core.bias_ih_l0[hidden_size:2 * hidden_size].fill_(1.0)

        self.land_encoder: nn.Linear | None = None
        self.context_projection: nn.Linear | None = None
        self.up_projection: nn.Linear | None = None
        if adapter_mode in {"conditional", "static_bias"}:
            self.land_encoder = nn.Linear(land_size, rank)
        if adapter_mode in {"conditional", "shared"}:
            self.context_projection = nn.Linear(dynamic_size + hidden_size, rank, bias=False)
        if adapter_mode != "none":
            self.up_projection = nn.Linear(rank, hidden_size, bias=False)
            nn.init.zeros_(self.up_projection.weight)

    def _validate(self, dynamic: Tensor, static: Tensor, land: Tensor | None) -> None:
        if dynamic.ndim != 3 or dynamic.shape[2] != self.dynamic_size or dynamic.shape[1] < 1:
            raise ValueError(f"dynamic must have shape [B, T>=1, {self.dynamic_size}].")
        if dynamic.shape[0] < 1:
            raise ValueError("Empty batches are not supported.")
        if static.shape != (dynamic.shape[0], self.static_size):
            raise ValueError(f"static must have shape [B, {self.static_size}].")
        if not torch.is_floating_point(dynamic) or not torch.is_floating_point(static):
            raise TypeError("Inputs must be floating-point tensors.")
        if static.device != dynamic.device or static.dtype != dynamic.dtype:
            raise ValueError("dynamic and static must share device and dtype.")
        if self.adapter_mode in {"conditional", "static_bias"}:
            if land is None or land.shape != (dynamic.shape[0], self.land_size):
                raise ValueError(f"land must have shape [B, {self.land_size}].")
            if land.device != dynamic.device or land.dtype != dynamic.dtype:
                raise ValueError("land must share dynamic's device and dtype.")

    def forward(
        self,
        dynamic: Tensor,
        static: Tensor,
        land: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Predict the last day of a causal window, with states initialized to zero."""
        self._validate(dynamic, static, land)
        batch, steps, _ = dynamic.shape
        h = dynamic.new_zeros(batch, self.hidden_size)
        c = dynamic.new_zeros(batch, self.hidden_size)
        hsize = self.hidden_size

        # Precompute the input affine transform without repeating static inputs.
        input_terms = F.linear(dynamic, self.core.weight_ih_l0[:, :self.dynamic_size])
        static_terms = F.linear(
            static, self.core.weight_ih_l0[:, self.dynamic_size:], self.core.bias_ih_l0
        )
        input_terms = input_terms + static_terms.unsqueeze(1)
        coefficients = None
        if self.land_encoder is not None:
            assert land is not None
            coefficients = torch.tanh(self.land_encoder(land))

        forget_values: list[Tensor] = []
        residual_values: list[Tensor] = []
        hidden_values: list[Tensor] = []
        for t in range(steps):
            gates = input_terms[:, t] + F.linear(h, self.core.weight_hh_l0, self.core.bias_hh_l0)
            i_logit, f_logit, g_logit, o_logit = gates.chunk(4, dim=-1)
            delta = torch.zeros_like(f_logit)
            if self.up_projection is not None:
                if self.adapter_mode == "static_bias":
                    assert coefficients is not None
                    low_rank = coefficients
                else:
                    assert self.context_projection is not None
                    low_rank = self.context_projection(torch.cat((dynamic[:, t], h), dim=-1))
                    if self.adapter_mode == "conditional":
                        assert coefficients is not None
                        low_rank = coefficients * low_rank
                delta = self.up_projection(low_rank)
            i = torch.sigmoid(i_logit)
            f = torch.sigmoid(f_logit + delta)
            g = torch.tanh(g_logit)
            o = torch.sigmoid(o_logit)
            c = f * c + i * g
            h = o * torch.tanh(c)
            if return_diagnostics:
                forget_values.append(f.detach())
                residual_values.append(delta.detach())
                hidden_values.append(h.detach())
        q_scaled = F.softplus(self.head(self.head_dropout(h)))
        if return_diagnostics:
            return q_scaled, {
                "forget_gate": torch.stack(forget_values, dim=1),
                "delta_f": torch.stack(residual_values, dim=1),
                "hidden_sequence": torch.stack(hidden_values, dim=1),
                "last_cell": c.detach(),
            }
        return q_scaled

    def fused_baseline(self, dynamic: Tensor, static: Tensor) -> Tensor:
        """Native PyTorch baseline with identical core/head; ignores the adapter."""
        if dynamic.ndim != 3 or static.shape != (dynamic.shape[0], self.static_size):
            raise ValueError("Invalid input shape.")
        inputs = torch.cat((dynamic, static[:, None].expand(-1, dynamic.shape[1], -1)), dim=-1)
        outputs, _ = self.core(inputs)
        return F.softplus(self.head(self.head_dropout(outputs[:, -1])))


def copy_base_weights(source: LandMemoryLSTM, target: LandMemoryLSTM) -> None:
    """Copy only LSTM/head tensors to give paired runs identical base initializations."""
    target.core.load_state_dict(source.core.state_dict())
    target.head.load_state_dict(source.head.state_dict())


def basin_scaled_mse(pred_mm: Tensor, target_mm: Tensor, sigma_mm: Tensor, epsilon: float = 0.1) -> Tensor:
    """Training loss for uniformly sampled basins; sigma uses SOURCE TRAINING Q only."""
    if pred_mm.ndim != 2 or pred_mm.shape[1] != 1 or target_mm.shape != pred_mm.shape:
        raise ValueError("pred_mm and target_mm must have the same [B, 1] shape.")
    if sigma_mm.ndim == 1:
        sigma_mm = sigma_mm.unsqueeze(-1)
    if sigma_mm.shape != pred_mm.shape:
        raise ValueError("sigma_mm must be [B] or [B, 1], not a broadcasted [B, B] matrix.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not torch.isfinite(pred_mm).all():
        raise FloatingPointError("Non-finite predictions.")
    valid = torch.isfinite(target_mm) & (target_mm >= 0)
    valid = valid & torch.isfinite(sigma_mm) & (sigma_mm >= 0)
    if not valid.any():
        raise ValueError("This batch has no valid source-training observations.")
    return (((pred_mm[valid] - target_mm[valid]) / (sigma_mm[valid] + epsilon)) ** 2).mean()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
