"""State-Space expert — Mamba-style S6 selective scan blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class S6Block(nn.Module):
    """Selective state-space block (Mamba-style S6).

    Projects input to inner dim, applies 1D causal conv, then selective SSM scan,
    and projects back.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_inner: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state

        self.norm = RMSNorm(d_model)

        # Input projections: one for the main path, one for the gate
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM parameters — input-dependent projections (selective)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Learnable A (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan (sequential for correctness, parallelisable with associative scan).

        Args:
            x:  (batch, seq_len, d_inner)
            dt: (batch, seq_len, d_inner) — step sizes
            B:  (batch, seq_len, d_state)
            C:  (batch, seq_len, d_state)

        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = B.shape[-1]

        # Discretize: A_bar = exp(A * dt), B_bar = B * dt
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        dt = F.softplus(dt)  # ensure positive

        # (batch, seq_len, d_inner, d_state)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq, d_inner, d_state)

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (batch, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        x = self.norm(x)

        # Input projections → main + gate
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, S, d_inner)

        # 1D causal conv
        x_main = x_main.transpose(1, 2)  # (B, d_inner, S)
        x_main = self.conv1d(x_main)[:, :, :S]  # causal: trim padding
        x_main = x_main.transpose(1, 2)  # (B, S, d_inner)
        x_main = F.silu(x_main)

        # Input-dependent SSM parameters
        x_ssm = self.x_proj(x_main)  # (B, S, d_state*2 + 1)
        B_param = x_ssm[:, :, :self.d_state]
        C_param = x_ssm[:, :, self.d_state:2 * self.d_state]
        dt_raw = x_ssm[:, :, -1:]  # (B, S, 1)
        dt = self.dt_proj(dt_raw)  # (B, S, d_inner)

        # Selective scan
        y = self.selective_scan(x_main, dt, B_param, C_param)

        # Skip connection with D
        y = y + x_main * self.D.unsqueeze(0).unsqueeze(0)

        # Gate
        y = y * F.silu(z)

        # Output
        y = self.dropout(self.out_proj(y))
        return residual + y


class StateSpaceExpert(nn.Module):
    """Stack of S6Blocks — Mamba-style state-space expert.

    Input:  (batch, seq_len, 384)
    Output: (batch, seq_len, 384)
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 384,
        d_inner: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                S6Block(d_model, d_inner, d_state, d_conv, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
