#!/usr/bin/env python3
"""Pre-train individual expert models as standalone language models.

Usage:
    python scripts/pretrain_experts.py --expert transformer
    python scripts/pretrain_experts.py --expert diffuser
    python scripts/pretrain_experts.py --expert state_space
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nano_think.config import (
    DataConfig,
    DiffuserConfig,
    ModelConfig,
    StateSpaceConfig,
    TrainDiffuserConfig,
    TrainExpertConfig,
    TransformerConfig,
)
from nano_think.data import build_dataloader
from nano_think.experts.diffuser import DiffuserExpert
from nano_think.experts.state_space import StateSpaceExpert
from nano_think.experts.transformer import TransformerExpert
from nano_think.train import (
    count_parameters,
    evaluate,
    get_lr,
    save_checkpoint,
    train_epoch,
)


class ExpertWithHead(nn.Module):
    """Wraps an expert module with a temporary LM head for pre-training."""

    def __init__(self, expert: nn.Module, embedding: nn.Embedding, d_model: int, vocab_size: int):
        super().__init__()
        self.embedding = embedding
        self.expert = expert
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.expert(x)
        return self.lm_head(x)


class DiffuserWithHead(nn.Module):
    """Wraps the diffuser expert for pre-training with a frozen encoder."""

    def __init__(
        self,
        diffuser: DiffuserExpert,
        encoder: nn.Module,
        embedding: nn.Embedding,
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.diffuser = diffuser

        # Freeze the encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.embedding.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns the diffusion training loss (MSE)."""
        with torch.no_grad():
            x = self.embedding(input_ids)
            target_embs = self.encoder(x)  # clean embeddings
        return self.diffuser.training_loss(target_embs)


def diffuser_loss_fn(model, batch, device, amp_dtype):
    """Custom loss function for diffuser pre-training."""
    input_ids = batch["input_ids"].to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
        loss = model(input_ids)
    return loss, {}


def build_expert(expert_name: str, model_cfg: ModelConfig, device: torch.device):
    """Build expert model with LM head for pre-training."""
    embedding = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)

    if expert_name == "transformer":
        cfg = TransformerConfig()
        expert = TransformerExpert(
            n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
            head_dim=cfg.head_dim, ffn_mult=cfg.ffn_mult, dropout=cfg.dropout,
            max_seq_len=model_cfg.max_seq_len, rope_base=cfg.rope_base,
        )
        model = ExpertWithHead(expert, embedding, cfg.d_model, model_cfg.vocab_size)
        train_cfg = TrainExpertConfig()

    elif expert_name == "state_space":
        cfg = StateSpaceConfig()
        expert = StateSpaceExpert(
            n_layers=cfg.n_layers, d_model=cfg.d_model, d_inner=cfg.d_inner,
            d_state=cfg.d_state, d_conv=cfg.d_conv, dropout=cfg.dropout,
        )
        model = ExpertWithHead(expert, embedding, cfg.d_model, model_cfg.vocab_size)
        train_cfg = TrainExpertConfig()

    elif expert_name == "diffuser":
        dcfg = DiffuserConfig()
        diffuser = DiffuserExpert(
            n_layers=dcfg.n_layers, d_model=dcfg.d_model, n_heads=dcfg.n_heads,
            head_dim=dcfg.head_dim, training_steps=dcfg.training_steps,
            inference_steps=dcfg.inference_steps, beta_start=dcfg.beta_start,
            beta_end=dcfg.beta_end, schedule=dcfg.schedule, dropout=dcfg.dropout,
        )

        # Build a frozen transformer encoder for target embeddings
        tcfg = TransformerConfig()
        encoder = TransformerExpert(
            n_layers=tcfg.n_layers, d_model=tcfg.d_model, n_heads=tcfg.n_heads,
            head_dim=tcfg.head_dim, ffn_mult=tcfg.ffn_mult, dropout=0.0,
            max_seq_len=model_cfg.max_seq_len,
        )

        # Load pre-trained transformer checkpoint if available
        ckpt_path = os.path.join(TrainExpertConfig().checkpoint_dir, "transformer_expert.pt")
        if os.path.exists(ckpt_path):
            print(f"Loading pre-trained transformer encoder from {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Extract just the expert weights
            expert_state = {}
            emb_state = {}
            for k, v in state["model"].items():
                if k.startswith("expert."):
                    expert_state[k[len("expert."):]] = v
                elif k.startswith("embedding."):
                    emb_state[k[len("embedding."):]] = v
            encoder.load_state_dict(expert_state, strict=False)
            embedding.load_state_dict(emb_state, strict=False)
        else:
            print(f"WARNING: No pre-trained transformer at {ckpt_path}. Using random encoder.")

        model = DiffuserWithHead(diffuser, encoder, embedding)
        train_cfg = TrainDiffuserConfig()

    else:
        raise ValueError(f"Unknown expert: {expert_name}")

    return model.to(device), train_cfg


def main():
    parser = argparse.ArgumentParser(description="Pre-train an expert model")
    parser.add_argument("--expert", type=str, required=True, choices=["transformer", "diffuser", "state_space"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # If user didn't explicitly set --data_dir, prefer data_prepared/ if it exists
    if args.data_dir == "data" and os.path.isdir("data_prepared"):
        print("Found data_prepared/ — using pre-chunked data")
        args.data_dir = "data_prepared"
    elif args.data_dir == "data":
        print("Warning: data_prepared/ not found. Run `python scripts/premunge.py` first to avoid truncation waste.")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Detect actual vocab size from tokenizer
    from nano_think.tokenizer import Tokenizer
    tok = Tokenizer()
    model_cfg = ModelConfig(vocab_size=tok.vocab_size)
    print(f"Vocab size: {model_cfg.vocab_size}")

    # Build model
    model, train_cfg = build_expert(args.expert, model_cfg, device)
    n_params = count_parameters(model, trainable_only=True)
    print(f"Expert: {args.expert} | Trainable parameters: {n_params:,}")

    # Data
    train_loader = build_dataloader(
        data_dir=args.data_dir,
        max_seq_len=train_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
        split="train",
        val_split=train_cfg.val_split,
        domain_balanced=True,
        num_workers=2,
    )
    val_loader = build_dataloader(
        data_dir=args.data_dir,
        max_seq_len=train_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
        split="val",
        val_split=train_cfg.val_split,
        domain_balanced=False,
        num_workers=2,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # AMP
    amp_dtype = torch.bfloat16 if train_cfg.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler() if (train_cfg.use_amp and device.type == "cuda" and amp_dtype == torch.float16) else None

    # LR schedule
    max_steps = len(train_loader) * train_cfg.max_epochs
    lr_fn = partial(get_lr, warmup_steps=train_cfg.warmup_steps, max_steps=max_steps, max_lr=train_cfg.lr)

    # Resume
    step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        from nano_think.train import load_checkpoint
        meta = load_checkpoint(args.resume, model, optimizer, scaler)
        step = meta["step"]
        start_epoch = meta["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}, step {step}")

    # Custom loss for diffuser
    loss_fn = diffuser_loss_fn if args.expert == "diffuser" else None

    # Checkpoint dir
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5

    print(f"\n{'='*60}")
    print(f"Pre-training {args.expert} expert")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg.max_epochs):
        train_loss, step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler_fn=lr_fn,
            step=step,
            epoch=epoch,
            device=device,
            grad_clip=train_cfg.grad_clip,
            log_interval=train_cfg.log_interval,
            scaler=scaler,
            amp_dtype=amp_dtype,
            use_amp=train_cfg.use_amp and device.type == "cuda",
            compute_loss_fn=loss_fn,
        )

        val_loss = evaluate(model, val_loader, device, amp_dtype=amp_dtype, compute_loss_fn=loss_fn)
        val_ppl = math.exp(val_loss) if val_loss < 100 else float("inf")

        print(f"\nEpoch {epoch} complete | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

        # Save checkpoint
        ckpt_path = str(ckpt_dir / f"{args.expert}_expert.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, step, epoch, val_loss, ckpt_path, scaler)
            print(f"  -> Best model saved to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")

        # Also save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            periodic_path = str(ckpt_dir / f"{args.expert}_expert_epoch{epoch}.pt")
            save_checkpoint(model, optimizer, step, epoch, val_loss, periodic_path, scaler)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_dir / f'{args.expert}_expert.pt'}")


if __name__ == "__main__":
    main()
