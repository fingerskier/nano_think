"""Hyperparameter configuration dataclasses for nano_think."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Shared model dimensions."""

    d_model: int = 384
    vocab_size: int = 32000
    max_seq_len: int = 512
    n_heads: int = 6
    head_dim: int = 64
    dropout: float = 0.1


@dataclass
class TransformerConfig:
    """Transformer expert hyperparameters."""

    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    head_dim: int = 64
    ffn_mult: int = 4  # FFN inner dim = d_model * ffn_mult = 1536
    dropout: float = 0.1
    rope_base: float = 10000.0


@dataclass
class DiffuserConfig:
    """Diffuser expert hyperparameters."""

    n_layers: int = 4
    d_model: int = 384
    n_heads: int = 6
    head_dim: int = 64
    training_steps: int = 100  # T for forward diffusion
    inference_steps: int = 10  # DDIM sampling steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"  # "linear" or "cosine"
    dropout: float = 0.1


@dataclass
class StateSpaceConfig:
    """State-Space (Mamba-style S6) expert hyperparameters."""

    n_layers: int = 6
    d_model: int = 384
    d_inner: int = 768  # 2x expansion
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # local convolution width
    dropout: float = 0.1


@dataclass
class MLAConfig:
    """Multi-Head Latent Attention + Router hyperparameters."""

    d_model: int = 384
    n_heads: int = 6
    head_dim: int = 64
    n_experts: int = 3
    kv_latent_dim: int | None = None  # None = no KV compression
    dropout: float = 0.1
    rope_base: float = 10000.0


@dataclass
class VectorStoreConfig:
    """FAISS vector store settings."""

    d_model: int = 384
    top_k: int = 4
    max_vectors: int = 100_000
    use_gpu: bool = False


@dataclass
class TrainExpertConfig:
    """Training hyperparameters for expert pre-training."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_epochs: int = 20
    batch_size: int = 32
    max_seq_len: int = 512
    grad_clip: float = 1.0
    val_split: float = 0.1
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"


@dataclass
class TrainDiffuserConfig(TrainExpertConfig):
    """Diffuser-specific training overrides."""

    lr: float = 1e-4
    warmup_steps: int = 500
    batch_size: int = 64


@dataclass
class TrainMLAConfig:
    """Training hyperparameters for MLA (full pipeline) training."""

    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_epochs: int = 30
    batch_size: int = 16
    grad_accum_steps: int = 4  # effective batch = 64
    max_seq_len: int = 512
    grad_clip: float = 1.0
    val_split: float = 0.1
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    use_amp: bool = True
    amp_dtype: str = "bf16"
    entropy_reg: float = 0.01  # router entropy regularisation weight
    transformer_ckpt: str = "checkpoints/transformer_expert.pt"
    diffuser_ckpt: str = "checkpoints/diffuser_expert.pt"
    state_space_ckpt: str = "checkpoints/state_space_expert.pt"


@dataclass
class DataConfig:
    """Data pipeline settings."""

    data_dir: str = "data"
    max_seq_len: int = 512
    tokenizer_name: str = "gpt2"
    domains: list[str] = field(
        default_factory=lambda: [
            "academic",
            "general",
            "code",
            "philosophy",
            "theology",
        ]
    )
    val_split: float = 0.1
    num_workers: int = 4
