# nano_think — PyTorch Implementation & Training Plan

## Phase 0: Project Setup

- [ ] Initialize Python package structure
  ```
  nano_think/
  ├── pyproject.toml
  ├── nano_think/
  │   ├── __init__.py
  │   ├── config.py          # dataclasses for all hyperparams
  │   ├── tokenizer.py       # BPE tokenizer wrapper
  │   ├── data.py            # Dataset / DataLoader utilities
  │   ├── experts/
  │   │   ├── __init__.py
  │   │   ├── transformer.py
  │   │   ├── diffuser.py
  │   │   └── state_space.py
  │   ├── mla.py             # Multi-Head Latent Attention + router
  │   ├── vector_store.py    # FAISS-backed retrieval module
  │   ├── model.py           # Full pipeline (embed → retrieve → MLA → experts → fuse → decode)
  │   └── train.py           # Training loops
  └── scripts/
      ├── pretrain_experts.py
      ├── train_mla.py
      └── generate.py
  ```
- [ ] Pin dependencies in `pyproject.toml`
  - `torch >= 2.2`
  - `faiss-gpu` (or `faiss-cpu`)
  - `transformers` (for tokenizer only)
  - `einops`
  - `wandb` (optional, for logging)
- [ ] Write a `config.py` with dataclass defaults
  - `d_model = 384`, `n_heads = 6`, `head_dim = 64`
  - `n_expert_layers = 6`, `ffn_mult = 4`, `vocab_size = 32000`
  - `top_k_vectors = 4`, `diffusion_steps = 10`
  - Per-expert and per-training-phase configs

---

## Phase 1: Data Pipeline

- [ ] Implement `data.py`
  - [ ] JSONL reader that streams from `data/` subdirectories
  - [ ] Tokenize texts using a BPE tokenizer (GPT-2 or train a custom 32k vocab on the corpus)
  - [ ] Pack/pad sequences to a fixed `max_seq_len` (e.g. 512)
  - [ ] Build `torch.utils.data.Dataset` and `DataLoader` with shuffling, batching
  - [ ] Domain-balanced sampler (proportional or equal sampling across domains)
- [ ] Validate data loading: confirm shapes, token distributions, no data leakage between train/val splits (90/10 split)

---

## Phase 2: Expert Modules

### 2a. Transformer Expert

- [ ] Implement `experts/transformer.py`
  - [ ] `TransformerBlock`: Pre-norm (RMSNorm), multi-head self-attention with RoPE, SwiGLU FFN
  - [ ] `TransformerExpert(nn.Module)`: Stack of 6 `TransformerBlock` layers
  - [ ] Causal attention mask for autoregressive generation
  - [ ] Input: `(batch, seq_len, 384)` → Output: `(batch, seq_len, 384)`
- [ ] Unit test: verify output shapes, causal masking, gradient flow

### 2b. Diffuser Expert

- [ ] Implement `experts/diffuser.py`
  - [ ] Noise schedule: linear or cosine β schedule, T=100 training steps
  - [ ] `DiffusionDenoiser(nn.Module)`: Small Transformer (4 layers, 384D) conditioned on timestep embedding
  - [ ] Forward process: `q(x_t | x_0)` — add noise to 384D embeddings
  - [ ] Training loss: MSE between predicted noise and actual noise (ε-prediction)
  - [ ] Inference: DDIM sampler with ~10 steps for fast generation
  - [ ] Input: `(batch, seq_len, 384)` noisy + timestep → Output: `(batch, seq_len, 384)` denoised
- [ ] Unit test: verify noise schedule, denoiser shapes, reconstruction quality on toy data

### 2c. State-Space Expert

- [ ] Implement `experts/state_space.py`
  - [ ] `S6Block`: Selective state-space layer (Mamba-style) with input-dependent Δ, B, C
  - [ ] Linear projection 384 → 768 (inner dim), 1D conv, selective scan, projection back to 384
  - [ ] `StateSpaceExpert(nn.Module)`: Stack of 6 `S6Block` layers with RMSNorm
  - [ ] Input: `(batch, seq_len, 384)` → Output: `(batch, seq_len, 384)`
- [ ] Unit test: verify shapes, autoregressive compatibility, gradient flow

---

## Phase 3: Vector Store

- [ ] Implement `vector_store.py`
  - [ ] Wrap FAISS `IndexFlatIP` (inner product on L2-normalized vectors)
  - [ ] `add(vectors: Tensor)` — normalize and insert into index
  - [ ] `search(query: Tensor, k: int) -> Tensor` — retrieve top-k nearest 384D vectors
  - [ ] `reset()` — clear the index
  - [ ] Handle GPU/CPU index selection based on device availability
- [ ] Integration test: insert random vectors, verify retrieval returns correct nearest neighbors

---

## Phase 4: Multi-Head Latent Attention & Router

- [ ] Implement `mla.py`
  - [ ] `MultiHeadLatentAttention(nn.Module)`
    - [ ] Standard 6-head self-attention (Q, K, V projections, 384D)
    - [ ] Optional KV compression: low-rank bottleneck for K and V (384 → latent → 384)
    - [ ] RoPE positional encoding
    - [ ] Pre-norm (RMSNorm) + residual connections
  - [ ] `ExpertRouter(nn.Module)`
    - [ ] Mean-pool attention output → linear 384→3 → softmax
    - [ ] Output: routing weights `(w_t, w_d, w_s)` per batch element
- [ ] Unit test: attention output shapes, router weights sum to 1, gradient flows to router

---

## Phase 5: Full Model Assembly

- [ ] Implement `model.py`
  - [ ] `NanoThink(nn.Module)` — full pipeline
    1. Token embedding (with tied weights for output head)
    2. Vector store lookup → prepend retrieved vectors to sequence
    3. MLA self-attention + routing
    4. Fork to three experts (parallel forward)
    5. Weighted fusion of expert outputs
    6. Vector store write (detached, no gradient)
    7. Output projection → vocab logits
  - [ ] `forward()` returns logits and routing weights (for logging/analysis)
  - [ ] `generate()` method for autoregressive inference with KV caching
- [ ] End-to-end test: random input → logits of correct shape, backward pass succeeds

---

## Phase 6: Pre-train Experts

Each expert is pre-trained independently as a standalone language model (with a temporary output head).

### 6a. Pre-train Transformer Expert

- [ ] Script: `scripts/pretrain_experts.py --expert transformer`
- [ ] Attach a temporary linear head: 384 → vocab_size
- [ ] Loss: cross-entropy next-token prediction
- [ ] Optimizer: AdamW, lr=3e-4, weight_decay=0.1, warmup 1000 steps, cosine decay
- [ ] Batch size: 32, seq_len: 512, epochs: ~10–20 (until val loss plateaus)
- [ ] Log train/val loss, perplexity per domain
- [ ] Save checkpoint: `checkpoints/transformer_expert.pt`

### 6b. Pre-train Diffuser Expert

- [ ] Script: `scripts/pretrain_experts.py --expert diffuser`
- [ ] Use frozen Transformer encoder to produce target 384D embeddings from text
- [ ] Loss: MSE noise prediction (standard diffusion training)
- [ ] Optimizer: AdamW, lr=1e-4, warmup 500 steps, cosine decay
- [ ] Batch size: 64, T=100 noise steps during training
- [ ] Log train/val MSE loss, sample reconstructions
- [ ] Save checkpoint: `checkpoints/diffuser_expert.pt`

### 6c. Pre-train State-Space Expert

- [ ] Script: `scripts/pretrain_experts.py --expert state_space`
- [ ] Attach a temporary linear head: 384 → vocab_size
- [ ] Loss: cross-entropy next-token prediction
- [ ] Optimizer: AdamW, lr=3e-4, weight_decay=0.1, warmup 1000 steps, cosine decay
- [ ] Batch size: 32, seq_len: 512, epochs: ~10–20
- [ ] Log train/val loss, perplexity per domain
- [ ] Save checkpoint: `checkpoints/state_space_expert.pt`

---

## Phase 7: Train MLA (Full Pipeline)

- [ ] Script: `scripts/train_mla.py`
- [ ] Load pre-trained expert checkpoints and **freeze** all expert parameters
- [ ] Trainable parameters: token embeddings, MLA attention, expert router, output head
- [ ] Loss: cross-entropy next-token prediction on fused output
- [ ] Auxiliary loss (optional): entropy regularization on router weights to encourage expert diversity
- [ ] Optimizer: AdamW, lr=1e-4, weight_decay=0.05, warmup 500 steps, cosine decay
- [ ] Batch size: 16 (full pipeline is memory-heavy with 3 expert forwards)
- [ ] Mixed precision: `torch.autocast` with bf16/fp16
- [ ] Gradient accumulation: effective batch size of 64
- [ ] Log: loss, perplexity, per-expert routing weights (mean, std), per-domain metrics
- [ ] Checkpointing: save full model every N steps
- [ ] Training duration: until val loss plateaus (monitor for 3+ epochs without improvement)

---

## Phase 8: Evaluation & Inference

- [ ] Implement `scripts/generate.py`
  - [ ] Autoregressive text generation with top-k / top-p sampling
  - [ ] Temperature control
  - [ ] Display expert routing weights per token (which expert is trusted most)
- [ ] Evaluate on held-out validation set
  - [ ] Perplexity per domain (academic, general, code, philosophy, theology)
  - [ ] Routing weight distribution analysis (do experts specialize by domain?)
- [ ] Qualitative evaluation
  - [ ] Generate samples across all domains
  - [ ] Compare full model vs individual experts
  - [ ] Analyze vector store retrieval relevance

---

## Phase 9: Optimization & Deployment

- [ ] Profile inference latency on target GPU
- [ ] Quantize to INT8 if needed for deployment
- [ ] Export model with `torch.compile()` for optimized inference
- [ ] Build simple CLI or API for local text generation
- [ ] Document usage instructions in README

---

## Suggested Order of Work

| Priority | Task                        | Depends On   |
|----------|-----------------------------|--------------|
| 1        | Phase 0: Project setup      | —            |
| 2        | Phase 1: Data pipeline      | Phase 0      |
| 3        | Phase 2a: Transformer       | Phase 0      |
| 4        | Phase 2c: State-Space       | Phase 0      |
| 5        | Phase 2b: Diffuser          | Phase 2a     |
| 6        | Phase 6a: Pretrain Transformer | Phase 1, 2a |
| 7        | Phase 6c: Pretrain State-Space | Phase 1, 2c |
| 8        | Phase 3: Vector store       | Phase 0      |
| 9        | Phase 4: MLA + Router       | Phase 0      |
| 10       | Phase 6b: Pretrain Diffuser | Phase 1, 2b, 6a |
| 11       | Phase 5: Full model         | Phase 2–4    |
| 12       | Phase 7: Train MLA          | Phase 5, 6   |
| 13       | Phase 8: Eval & inference   | Phase 7      |
| 14       | Phase 9: Optimization       | Phase 8      |

Phases 2a, 2c, 3, and 4 can be developed in parallel. Expert pre-training (6a, 6c) can also run concurrently once their respective modules and the data pipeline are ready.
