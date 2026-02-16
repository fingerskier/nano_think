"""Transformer expert — 6-layer autoregressive decoder with RoPE, SwiGLU, RMSNorm."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Pre-compute complex-valued RoPE frequencies."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensor.

    Args:
        x: (batch, n_heads, seq_len, head_dim)
        freqs: (seq_len, head_dim//2) complex
    """
    # Reshape to pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[2]].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, hd//2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: gate * swish(gate_proj) * up_proj."""

    def __init__(self, d_model: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = d_model * ffn_mult
        self.w_gate = nn.Linear(d_model, inner, bias=False)
        self.w_up = nn.Linear(d_model, inner, bias=False)
        self.w_down = nn.Linear(inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with multi-head self-attention + SwiGLU FFN."""

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        head_dim: int = 64,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.ffn = SwiGLU(d_model, ffn_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.attn_norm(x)
        B, S, _ = h.shape

        q = rearrange(self.wq(h), "b s (nh hd) -> b nh s hd", nh=self.n_heads)
        k = rearrange(self.wk(h), "b s (nh hd) -> b nh s hd", nh=self.n_heads)
        v = rearrange(self.wv(h), "b s (nh hd) -> b nh s hd", nh=self.n_heads)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b nh s hd -> b s (nh hd)")
        out = self.resid_drop(self.wo(out))

        x = x + out

        # Pre-norm FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class TransformerExpert(nn.Module):
    """Stack of TransformerBlock layers — autoregressive decoder expert.

    Input:  (batch, seq_len, 384)
    Output: (batch, seq_len, 384)
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 384,
        n_heads: int = 6,
        head_dim: int = 64,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, head_dim, ffn_mult, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

        # Pre-compute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(head_dim, max_seq_len * 2, rope_base),
            persistent=False,
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape
        if mask is None:
            mask = self._causal_mask(S, x.device)

        for layer in self.layers:
            x = layer(x, self.rope_freqs, mask)

        return self.norm(x)
