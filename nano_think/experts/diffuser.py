"""Diffuser expert — discrete diffusion model in 384D embedding space."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) integer timesteps -> (batch, d_model)"""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class DenoiserBlock(nn.Module):
    """Transformer block with timestep conditioning for the denoiser."""

    def __init__(self, d_model: int = 384, n_heads: int = 6, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Timestep modulation (scale + shift for adaptive layer norm)
        self.t_proj = nn.Linear(d_model, d_model * 2)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        t_emb: (batch, d_model) timestep embedding
        """
        # Timestep-conditioned adaptive norm
        scale, shift = self.t_proj(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift

        B, S, _ = h.shape
        q = rearrange(self.wq(h), "b s (nh hd) -> b nh s hd", nh=self.n_heads)
        k = rearrange(self.wk(h), "b s (nh hd) -> b nh s hd", nh=self.n_heads)
        v = rearrange(self.wv(h), "b s (nh hd) -> b nh s hd", nh=self.n_heads)

        attn_scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * attn_scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b nh s hd -> b s (nh hd)")
        out = self.wo(out)
        x = x + out

        x = x + self.ffn(self.norm2(x))
        return x


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from 'Improved DDPM'."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    alphas_bar = torch.cos((steps + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(0, 0.999).float()


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class DiffuserExpert(nn.Module):
    """Discrete diffusion expert operating in 384D embedding space.

    Training: ε-prediction (noise prediction) with MSE loss.
    Inference: DDIM sampling with configurable number of steps.

    Input:  (batch, seq_len, 384) — clean embeddings
    Output: (batch, seq_len, 384) — denoised embeddings
    """

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 384,
        n_heads: int = 6,
        head_dim: int = 64,
        training_steps: int = 100,
        inference_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.T = training_steps
        self.inference_steps = inference_steps

        # Noise schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(training_steps)
        else:
            betas = linear_beta_schedule(training_steps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(d_model)

        # Denoiser: small Transformer
        self.layers = nn.ModuleList(
            [DenoiserBlock(d_model, n_heads, head_dim, dropout) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run the denoiser to predict the noise component."""
        t_emb = self.time_embed(t)

        h = x_t
        for layer in self.layers:
            h = layer(h, t_emb)
        h = self.final_norm(h)
        return self.out_proj(h)

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute ε-prediction MSE loss for a batch of clean embeddings."""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)
        x_t, noise = self.q_sample(x_0, t)
        predicted = self.predict_noise(x_t, t)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def ddim_sample(self, x_T: torch.Tensor) -> torch.Tensor:
        """DDIM deterministic sampling for inference."""
        # Sub-select timesteps for faster inference
        step_size = self.T // self.inference_steps
        timesteps = list(range(self.T - 1, -1, -step_size))

        x = x_T
        for i, t_val in enumerate(timesteps):
            t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.long)
            pred_noise = self.predict_noise(x, t)

            alpha_t = self.alphas_cumprod[t_val]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

            # DDIM update (η=0, deterministic)
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: add noise then denoise via DDIM.

        During training, use training_loss() instead.
        At inference, this produces a denoised version of the input.
        """
        if self.training:
            # During training, just return the input — use training_loss() for the loss
            return x

        # Inference: light noise + denoise for refinement
        noise = torch.randn_like(x)
        # Start from a partially noised version (not full T)
        start_t = self.inference_steps
        t = torch.full((x.shape[0],), min(start_t, self.T - 1), device=x.device, dtype=torch.long)
        x_noisy, _ = self.q_sample(x, t, noise)
        return self.ddim_sample(x_noisy)
