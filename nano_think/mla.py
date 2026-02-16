"""Multi-Head Latent Attention (MLA) and Expert Router."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .experts.transformer import RMSNorm, precompute_rope_freqs, apply_rope


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention with optional KV compression.

    Serves two purposes:
    1. Contextual encoding — self-attention over augmented context
    2. Expert routing — produces routing weights via ExpertRouter
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        head_dim: int = 64,
        kv_latent_dim: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.kv_latent_dim = kv_latent_dim

        self.norm = RMSNorm(d_model)

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)

        if kv_latent_dim is not None:
            # KV compression: project down then back up (DeepSeek-V2 style)
            self.kv_down = nn.Linear(d_model, kv_latent_dim, bias=False)
            self.wk = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
            self.wv = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
        else:
            self.kv_down = None
            self.wk = nn.Linear(d_model, n_heads * head_dim, bias=False)
            self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)

        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # RoPE
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(head_dim, max_seq_len * 2, rope_base),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — augmented context (embeddings + retrieved vecs)
            mask: optional attention mask

        Returns:
            context: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        B, S, _ = x.shape

        q = rearrange(self.wq(x), "b s (nh hd) -> b nh s hd", nh=self.n_heads)

        kv_input = self.kv_down(x) if self.kv_down is not None else x
        k = rearrange(self.wk(kv_input), "b s (nh hd) -> b nh s hd", nh=self.n_heads)
        v = rearrange(self.wv(kv_input), "b s (nh hd) -> b nh s hd", nh=self.n_heads)

        # Apply RoPE
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b nh s hd -> b s (nh hd)")
        out = self.resid_drop(self.wo(out))

        return residual + out


class ExpertRouter(nn.Module):
    """Routes to experts via mean-pooled MLA output → softmax over 3 experts.

    Output: routing weights (w_t, w_d, w_s) — one set per batch element.
    """

    def __init__(self, d_model: int = 384, n_experts: int = 3):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, context: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            context: (batch, seq_len, d_model) — MLA output
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            weights: (batch, n_experts) — softmax routing weights
        """
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            pooled = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = context.mean(dim=1)  # (batch, d_model)

        logits = self.router(pooled)  # (batch, n_experts)
        return F.softmax(logits, dim=-1)
