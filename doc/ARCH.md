# nano_think Architecture

## Overview

nano_think is a Mixture-of-Experts (MoE) language model built around **Multi-Head Latent Attention (MLA)** routing. Three independently pre-trained expert modules—Transformer, Diffuser, and State-Space—operate in a shared 384-dimensional embedding space. A learnable MLA layer routes inputs to experts and fuses their outputs via weighted combination. A vector store provides retrieval-augmented context at inference time.

The design targets quick, local inference on consumer GPUs.

---

## Embedding Space

All modules share a **384-dimensional** representation. This keeps parameter counts low and ensures expert outputs can be directly combined without projection layers.

| Component          | Input dim | Hidden dim | Output dim |
|--------------------|-----------|------------|------------|
| Token embedding    | vocab_size| —          | 384        |
| Transformer expert | 384       | TBD        | 384        |
| Diffuser expert    | 384       | TBD        | 384        |
| State-Space expert | 384       | TBD        | 384        |
| MLA router         | 384       | TBD        | 3 (weights)|
| Vector store       | 384       | —          | 384        |

---

## Pipeline

```
Input tokens
    │
    ▼
┌──────────────────────┐
│  Token Embedding     │  vocab → 384D
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vector Store Lookup │  Retrieve top-k nearest 384D vectors
└──────────┬───────────┘  and concatenate to context
           │
           ▼
┌──────────────────────┐
│  Multi-Head Latent   │  Compute attention over context;
│  Attention (MLA)     │  produce expert routing weights (w_t, w_d, w_s)
└──────────┬───────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
   ┌───┐ ┌───┐ ┌───┐
   │ T │ │ D │ │ S │    Experts generate in parallel
   └─┬─┘ └─┬─┘ └─┬─┘
     │     │     │
     ▼     ▼     ▼
┌──────────────────────┐
│  Weighted Fusion     │  output = w_t·T + w_d·D + w_s·S
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vector Store Write  │  Encode output → 384D, store for future retrieval
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Output Head         │  384D → vocab logits → tokens
└──────────────────────┘
```

---

## Component Details

### 1. Token Embedding

Standard learned embedding table mapping token IDs to 384D vectors. A shared embedding is used for both input and output (tied weights).

- **Tokenizer**: BPE (e.g. GPT-2 tokenizer or a custom small-vocab BPE trained on the data corpus).
- **Positional encoding**: Rotary Position Embeddings (RoPE) applied inside each expert, so the embedding layer itself is position-agnostic.

### 2. Vector Store

A dense retrieval index over 384D vectors using FAISS (`IndexFlatIP` or `IndexIVFFlat`).

- **Write path**: After each forward pass, the fused output representation is L2-normalized and inserted into the index.
- **Read path**: The input embedding is used as a query to retrieve the top-k (e.g. k=4) nearest neighbors. Retrieved vectors are prepended to the context sequence before the MLA step.
- **Purpose**: Gives the model a persistent, growing memory that augments its context window without increasing sequence length at training time.

### 3. Multi-Head Latent Attention (MLA)

The MLA block serves two purposes:

1. **Contextual encoding** — standard multi-head self-attention over the augmented context (input embeddings + retrieved vectors).
2. **Expert routing** — a lightweight projection from the attention output to 3 scalar weights, passed through softmax to produce a probability distribution over experts.

Architecture:
```
x ∈ R^(seq_len × 384)

  ┌─ Q = x · W_q    (384 → 384)
  ├─ K = x · W_k    (384 → 384)
  └─ V = x · W_v    (384 → 384)

attn_out = softmax(Q·K^T / √d_k) · V      # standard MHA
context  = LayerNorm(attn_out + x)          # residual + norm

router_logits = MeanPool(context) · W_r     # (384 → 3)
weights = softmax(router_logits)            # (w_t, w_d, w_s)
```

- **Heads**: 6 (head dim = 64)
- **Latent bottleneck** (optional): Compress K and V through a low-rank projection (384 → latent_dim → 384) to reduce KV-cache memory, following DeepSeek-V2 style MLA.

### 4. Transformer Expert

A small autoregressive Transformer decoder.

- **Layers**: 6
- **Hidden dim**: 384
- **FFN inner dim**: 1536 (4× expansion)
- **Attention heads**: 6
- **Positional encoding**: RoPE
- **Activation**: SwiGLU
- **Role**: Primary sequence-to-sequence generator. Handles the bulk of language modeling.

### 5. Diffuser Expert

A discrete diffusion model operating in the 384D embedding space.

- **Forward process**: Adds Gaussian noise to token embeddings over T diffusion steps.
- **Reverse network**: A small U-Net or Transformer-based denoiser (4 layers, 384D) that predicts the clean embedding from noisy input + timestep embedding.
- **Inference**: Iterative denoising (e.g. T=10 steps with DDIM scheduling) to produce a 384D output sequence.
- **Role**: Provides a complementary generation mechanism that can capture global coherence and handle non-autoregressive generation patterns.

### 6. State-Space Expert

A structured state-space sequence model (Mamba-style S4/S6).

- **Layers**: 6
- **State dim**: 384
- **SSM inner dim**: 768 (2× expansion)
- **Activation**: SiLU
- **Selective scan**: Input-dependent state transitions (Mamba-style).
- **Role**: Efficient long-range dependency modeling with linear complexity in sequence length. Complements the Transformer's quadratic attention with better scaling on longer sequences.

### 7. Weighted Fusion

Expert outputs are combined via the MLA-produced routing weights:

```
output = w_t · expert_transformer(context) +
         w_d · expert_diffuser(context) +
         w_s · expert_statespace(context)
```

All expert outputs are 384D sequences of the same length, so fusion is elementwise weighted addition.

### 8. Output Head

A linear projection from 384D to vocabulary size, with tied weights to the input embedding matrix. Produces logits for next-token prediction (or full-sequence prediction for the diffuser path).

---

## Parameter Estimate

| Component          | Approx. Parameters |
|--------------------|--------------------|
| Token embedding    | vocab × 384       |
| MLA (6-head)       | ~0.9M              |
| Transformer expert | ~7M                |
| Diffuser expert    | ~5M                |
| State-Space expert | ~5M                |
| Output head (tied) | 0 (shared)         |
| **Total**          | **~18M + vocab×384** |

With a 32k vocabulary: ~18M + 12.3M ≈ **~30M parameters**.

---

## Training Data

| Domain     | Dataset           | Samples | Description                        |
|------------|-------------------|---------|------------------------------------|
| Academic   | arxiv_abstracts   | 10,000  | Scientific abstracts (ML, physics) |
| Academic   | ml_arxiv          | 10,000  | Machine learning papers            |
| General    | tinystories       | 10,000  | Short children's stories           |
| General    | wikitext          | 10,000  | Wikipedia articles                 |
| Code       | codeparrot        | 5,000   | Python source code                 |
| Philosophy | stanford_plato    | 1,000   | Stanford Encyclopedia entries      |
| Theology   | openbible         | 10,000  | Bible verses (KJV)                 |

**Total**: ~56,000 samples, ~116 MB

---

## Design Rationale

- **Why MoE?** Different expert architectures have complementary strengths: Transformers excel at local attention patterns, SSMs handle long-range dependencies efficiently, and diffusion models capture global structure. Letting a learned router combine them adaptively should outperform any single architecture at this scale.
- **Why 384D?** Small enough for consumer GPU inference, large enough for meaningful representation. Matches common small-model embedding sizes (e.g. all-MiniLM-L6-v2).
- **Why retrieval augmentation?** Extends effective context without increasing compute at each step. Lets the model "remember" past generations and build on them.
- **Why freeze experts during MLA training?** Prevents catastrophic forgetting of pre-trained capabilities. The MLA only needs to learn when to trust each expert—a much simpler optimization surface.
