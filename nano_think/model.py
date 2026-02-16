"""NanoThink — full MoE pipeline with MLA routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DiffuserConfig,
    MLAConfig,
    ModelConfig,
    StateSpaceConfig,
    TransformerConfig,
    VectorStoreConfig,
)
from .experts.diffuser import DiffuserExpert
from .experts.state_space import StateSpaceExpert
from .experts.transformer import TransformerExpert
from .mla import ExpertRouter, MultiHeadLatentAttention
from .vector_store import VectorStore


class NanoThink(nn.Module):
    """Full nano_think pipeline.

    Pipeline:
        1. Token embedding (with tied weights for output head)
        2. Vector store lookup → prepend retrieved vectors to sequence
        3. MLA self-attention + routing
        4. Fork to three experts (parallel forward)
        5. Weighted fusion of expert outputs
        6. Vector store write (detached, no gradient)
        7. Output projection → vocab logits

    forward() returns logits and routing weights.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        transformer_cfg: TransformerConfig | None = None,
        diffuser_cfg: DiffuserConfig | None = None,
        ssm_cfg: StateSpaceConfig | None = None,
        mla_cfg: MLAConfig | None = None,
        vs_cfg: VectorStoreConfig | None = None,
    ):
        super().__init__()

        mc = model_cfg or ModelConfig()
        tc = transformer_cfg or TransformerConfig()
        dc = diffuser_cfg or DiffuserConfig()
        sc = ssm_cfg or StateSpaceConfig()
        mla = mla_cfg or MLAConfig()
        vc = vs_cfg or VectorStoreConfig()

        self.d_model = mc.d_model
        self.vocab_size = mc.vocab_size
        self.max_seq_len = mc.max_seq_len
        self.top_k = vc.top_k

        # 1. Token embedding (shared with output head)
        self.token_emb = nn.Embedding(mc.vocab_size, mc.d_model)
        self.emb_drop = nn.Dropout(mc.dropout)

        # 2. Vector store (non-parametric, not an nn.Module)
        self.vector_store = VectorStore(d_model=vc.d_model, use_gpu=vc.use_gpu)

        # 3. MLA + Router
        self.mla = MultiHeadLatentAttention(
            d_model=mla.d_model,
            n_heads=mla.n_heads,
            head_dim=mla.head_dim,
            kv_latent_dim=mla.kv_latent_dim,
            dropout=mla.dropout,
            max_seq_len=mc.max_seq_len + vc.top_k,  # account for prepended retrieved vecs
            rope_base=mla.rope_base,
        )
        self.router = ExpertRouter(d_model=mla.d_model, n_experts=mla.n_experts)

        # 4. Expert modules
        self.transformer_expert = TransformerExpert(
            n_layers=tc.n_layers,
            d_model=tc.d_model,
            n_heads=tc.n_heads,
            head_dim=tc.head_dim,
            ffn_mult=tc.ffn_mult,
            dropout=tc.dropout,
            max_seq_len=mc.max_seq_len,
            rope_base=tc.rope_base,
        )
        self.diffuser_expert = DiffuserExpert(
            n_layers=dc.n_layers,
            d_model=dc.d_model,
            n_heads=dc.n_heads,
            head_dim=dc.head_dim,
            training_steps=dc.training_steps,
            inference_steps=dc.inference_steps,
            beta_start=dc.beta_start,
            beta_end=dc.beta_end,
            schedule=dc.schedule,
            dropout=dc.dropout,
        )
        self.state_space_expert = StateSpaceExpert(
            n_layers=sc.n_layers,
            d_model=sc.d_model,
            d_inner=sc.d_inner,
            d_state=sc.d_state,
            d_conv=sc.d_conv,
            dropout=sc.dropout,
        )

        # 7. Output head (tied to embedding)
        self.out_proj = nn.Linear(mc.d_model, mc.vocab_size, bias=False)
        self.out_proj.weight = self.token_emb.weight  # weight tying

    def _retrieve_and_prepend(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Retrieve top-k vectors and prepend to the sequence."""
        B, S, D = x.shape

        # Query with mean-pooled embeddings
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()
            query = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            query = x.mean(dim=1)  # (B, D)

        retrieved = self.vector_store.search(query, k=self.top_k)  # (B, k, D)

        # Prepend retrieved vectors to sequence
        x = torch.cat([retrieved, x], dim=1)  # (B, k+S, D)

        if attention_mask is not None:
            # Retrieved vectors always attend
            ret_mask = torch.ones(B, self.top_k, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([ret_mask, attention_mask], dim=1)

        return x, attention_mask

    def _strip_retrieved(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the prepended retrieved vectors from the sequence."""
        return x[:, self.top_k :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_vector_store: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for pad
            use_vector_store: whether to use retrieval augmentation (off during early training)

        Returns:
            dict with keys:
                logits: (batch, seq_len, vocab_size)
                router_weights: (batch, 3) — expert routing weights
        """
        # 1. Token embedding
        x = self.emb_drop(self.token_emb(input_ids))  # (B, S, D)
        original_seq_len = x.shape[1]

        # 2. Vector store lookup (optional)
        if use_vector_store and self.vector_store.size > 0:
            x, attention_mask = self._retrieve_and_prepend(x, attention_mask)

        # Build causal mask for MLA
        S_total = x.shape[1]
        causal = torch.tril(torch.ones(S_total, S_total, device=x.device))
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # 3. MLA self-attention
        context = self.mla(x, mask=causal)

        # Strip retrieved prefixes before passing to experts
        if use_vector_store and self.vector_store.size > 0:
            context_for_experts = self._strip_retrieved(context)
            # Also compute router on stripped context
            router_mask = attention_mask[:, self.top_k:] if attention_mask is not None else None
        else:
            context_for_experts = context
            router_mask = attention_mask

        # Router weights
        router_weights = self.router(context_for_experts, router_mask)  # (B, 3)

        # 4. Expert forward passes
        out_t = self.transformer_expert(context_for_experts)
        out_d = self.diffuser_expert(context_for_experts)
        out_s = self.state_space_expert(context_for_experts)

        # 5. Weighted fusion
        w = router_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, 3, 1)
        stacked = torch.stack([out_t, out_d, out_s], dim=2)  # (B, S, 3, D)
        fused = (stacked * w).sum(dim=2)  # (B, S, D)

        # 6. Vector store write (detached, no gradient)
        if use_vector_store:
            with torch.no_grad():
                mean_repr = fused.mean(dim=1)  # (B, D)
                self.vector_store.add(mean_repr)

        # 7. Output logits
        logits = self.out_proj(fused)  # (B, S, vocab)

        return {"logits": logits, "router_weights": router_weights}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_vector_store: bool = False,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.max_seq_len :]

            outputs = self.forward(idx_cond, use_vector_store=use_vector_store)
            logits = outputs["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
