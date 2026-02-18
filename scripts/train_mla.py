#!/usr/bin/env python3
"""Train the MLA (Multi-Head Latent Attention) pipeline with frozen experts.

Loads pre-trained expert checkpoints, freezes them, and trains:
  - Token embeddings
  - MLA attention
  - Expert router
  - Output head

Usage:
    python scripts/train_mla.py
    python scripts/train_mla.py --data_dir data --resume checkpoints/nanothink_mla.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nano_think.config import (
    DiffuserConfig,
    MLAConfig,
    ModelConfig,
    StateSpaceConfig,
    TrainMLAConfig,
    TransformerConfig,
    VectorStoreConfig,
)
from nano_think.data import build_dataloader
from nano_think.model import NanoThink
from nano_think.train import (
    count_parameters,
    evaluate,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
)


def load_expert_weights(model: NanoThink, cfg: TrainMLAConfig, device: torch.device) -> None:
    """Load pre-trained expert checkpoints into the full model."""
    expert_map = {
        "transformer": (cfg.transformer_ckpt, model.transformer_expert, "expert."),
        "state_space": (cfg.state_space_ckpt, model.state_space_expert, "expert."),
        "diffuser": (cfg.diffuser_ckpt, model.diffuser_expert, "diffuser."),
    }

    for name, (ckpt_path, expert_module, prefix) in expert_map.items():
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {name} checkpoint not found at {ckpt_path}, using random init")
            continue

        print(f"Loading {name} expert from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        expert_state = {}
        for k, v in state["model"].items():
            if k.startswith(prefix):
                expert_state[k[len(prefix):]] = v

        if expert_state:
            expert_module.load_state_dict(expert_state, strict=False)
            print(f"  Loaded {len(expert_state)} weight tensors for {name}")
        else:
            print(f"  WARNING: No weights with prefix '{prefix}' found in {ckpt_path}")

    # Also load embedding weights from transformer checkpoint if available
    if os.path.exists(cfg.transformer_ckpt):
        state = torch.load(cfg.transformer_ckpt, map_location="cpu", weights_only=False)
        emb_state = {}
        for k, v in state["model"].items():
            if k.startswith("embedding."):
                emb_state[k[len("embedding."):]] = v
        if emb_state:
            model.token_emb.load_state_dict(emb_state, strict=False)
            print("  Loaded embedding weights from transformer checkpoint")


def freeze_experts(model: NanoThink) -> None:
    """Freeze all expert parameters — only train embeddings, MLA, router, output head."""
    for expert in [model.transformer_expert, model.diffuser_expert, model.state_space_expert]:
        for p in expert.parameters():
            p.requires_grad = False


def mla_loss_fn(model, batch, device, amp_dtype, entropy_reg: float = 0.01):
    """Cross-entropy loss with optional entropy regularisation on router weights."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        router_weights = outputs["router_weights"]

        # Primary loss: cross-entropy
        ce_loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Auxiliary loss: encourage router diversity (penalise low entropy)
        router_entropy = -(router_weights * torch.log(router_weights + 1e-8)).sum(dim=-1).mean()
        # We want HIGH entropy → negate for penalty
        entropy_penalty = -entropy_reg * router_entropy

        loss = ce_loss + entropy_penalty

    extra = {
        "ce_loss": ce_loss.item(),
        "router_entropy": router_entropy.item(),
        "router_weights_mean": router_weights.mean(dim=0).detach().cpu().tolist(),
    }
    return loss, extra


def main():
    parser = argparse.ArgumentParser(description="Train MLA pipeline with frozen experts")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint path (omit path to auto-detect)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_vector_store", action="store_true", default=False)
    args = parser.parse_args()

    # Auto-detect checkpoint when --resume is used without a path
    if args.resume == "auto":
        import glob
        ckpt_dir = TrainMLAConfig().checkpoint_dir
        name = "nanothink_mla"
        candidates = []

        # Priority 1: interrupted checkpoint
        interrupted = os.path.join(ckpt_dir, f"{name}_interrupted.pt")
        if os.path.exists(interrupted):
            candidates.append(interrupted)

        # Priority 2: newest epoch checkpoint
        epoch_pattern = os.path.join(ckpt_dir, f"{name}_epoch*.pt")
        epoch_files = sorted(glob.glob(epoch_pattern), key=os.path.getmtime, reverse=True)
        candidates.extend(epoch_files)

        # Priority 3: best checkpoint
        best = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(best):
            candidates.append(best)

        if candidates:
            args.resume = candidates[0]
            print(f"Auto-detected checkpoint: {args.resume}")
        else:
            parser.error(f"--resume: no checkpoints found in {ckpt_dir}/")

    # If user didn't explicitly set --data_dir, prefer data_prepared/ if it exists
    if args.data_dir == "data" and os.path.isdir("data_prepared"):
        print("Found data_prepared/ — using pre-chunked data")
        args.data_dir = "data_prepared"
    elif args.data_dir == "data":
        print("Warning: data_prepared/ not found. Run `python scripts/premunge.py` first to avoid truncation waste.")

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

    train_cfg = TrainMLAConfig()

    # Build full model
    model = NanoThink(
        model_cfg=model_cfg,
        transformer_cfg=TransformerConfig(),
        diffuser_cfg=DiffuserConfig(),
        ssm_cfg=StateSpaceConfig(),
        mla_cfg=MLAConfig(),
        vs_cfg=VectorStoreConfig(),
    ).to(device)

    # Load expert checkpoints and freeze
    load_expert_weights(model, train_cfg, device)
    freeze_experts(model)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")

    # Data
    train_loader = build_dataloader(
        data_dir=args.data_dir,
        max_seq_len=train_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
        split="train",
        val_split=train_cfg.val_split,
        domain_balanced=True,
    )
    val_loader = build_dataloader(
        data_dir=args.data_dir,
        max_seq_len=train_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
        split="val",
        val_split=train_cfg.val_split,
        domain_balanced=False,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Optimizer — only trainable params
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
    max_steps = (len(train_loader) // train_cfg.grad_accum_steps) * train_cfg.max_epochs
    lr_fn = partial(get_lr, warmup_steps=train_cfg.warmup_steps, max_steps=max_steps, max_lr=train_cfg.lr)

    # Loss function with entropy regularisation
    loss_fn = partial(mla_loss_fn, entropy_reg=train_cfg.entropy_reg)

    # Resume
    step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        meta = load_checkpoint(args.resume, model, optimizer, scaler)
        step = meta["step"]
        start_epoch = meta["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}, step {step}")

    # Checkpoint dir
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5

    print(f"\n{'='*60}")
    print(f"Training MLA pipeline (experts frozen)")
    print(f"{'='*60}\n")

    try:
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
                grad_accum_steps=train_cfg.grad_accum_steps,
                compute_loss_fn=loss_fn,
            )

            val_loss = evaluate(model, val_loader, device, amp_dtype=amp_dtype, compute_loss_fn=loss_fn)
            val_ppl = math.exp(min(val_loss, 20))

            print(f"\nEpoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

            # Save
            ckpt_path = str(ckpt_dir / "nanothink_mla.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, step, epoch, val_loss, ckpt_path, scaler)
                print(f"  -> Best model saved to {ckpt_path}")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{patience})")

            if (epoch + 1) % 5 == 0:
                periodic_path = str(ckpt_dir / f"nanothink_mla_epoch{epoch}.pt")
                save_checkpoint(model, optimizer, step, epoch, val_loss, periodic_path, scaler)

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {epoch}, step {step}. Saving checkpoint...")
        interrupt_path = str(ckpt_dir / "nanothink_mla_interrupted.pt")
        save_checkpoint(model, optimizer, step, epoch, best_val_loss, interrupt_path, scaler)
        print(f"Checkpoint saved to {interrupt_path}")
        print(f"Resume with: python scripts/train_mla.py --resume")

    print(f"\nMLA training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'nanothink_mla.pt'}")


if __name__ == "__main__":
    main()
