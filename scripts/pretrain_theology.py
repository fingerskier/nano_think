#!/usr/bin/env python3
"""Theology-domain pre-training of the transformer expert.

Trains the transformer expert exclusively on theology data (Bible verses),
then evaluates domain-specific quality with perplexity, generation samples,
and cross-domain comparisons.

Usage:
    python scripts/pretrain_theology.py
    python scripts/pretrain_theology.py --max_epochs 10 --batch_size 16
    python scripts/pretrain_theology.py --eval_only --checkpoint checkpoints/theology_transformer.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nano_think.config import ModelConfig, TrainExpertConfig, TransformerConfig
from nano_think.data import build_dataloader
from nano_think.experts.transformer import TransformerExpert
from nano_think.tokenizer import Tokenizer
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
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.expert(x)
        return self.lm_head(x)


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, amp_dtype):
    """Compute perplexity on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(input_ids, attention_mask=mask)
            if isinstance(logits, dict):
                logits = logits["logits"]
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device, max_new_tokens=100, temperature=0.8, top_k=40):
    """Generate text completions from the model."""
    model.eval()
    results = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            seq = input_ids[:, -511:]

            logits = model(seq)
            if isinstance(logits, dict):
                logits = logits["logits"]

            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == tokenizer.eos_id:
                break

        generated = tokenizer.decode(input_ids[0].tolist())
        results.append({"prompt": prompt, "generated": generated})

    return results


def run_evaluation(model, tokenizer, device, amp_dtype, data_dir="data"):
    """Full evaluation suite for the theology-trained model."""
    results = {}

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # 1. Per-domain perplexity
    print("\n--- Per-Domain Perplexity ---")
    domains = ["theology", "philosophy", "general", "academic", "code"]
    domain_ppls = {}

    for domain in domains:
        domain_dir = Path(data_dir) / domain
        if not domain_dir.exists() or not list(domain_dir.glob("*.jsonl")):
            print(f"  {domain:12s}: [no data]")
            continue

        try:
            loader = build_dataloader(
                data_dir=data_dir,
                max_seq_len=512,
                batch_size=16,
                domains=[domain],
                split="val",
                val_split=0.1,
                domain_balanced=False,
            )
            if len(loader) == 0:
                print(f"  {domain:12s}: [no val batches]")
                continue

            avg_loss, ppl = evaluate_perplexity(model, loader, device, amp_dtype)
            domain_ppls[domain] = {"loss": avg_loss, "perplexity": ppl}
            print(f"  {domain:12s}: loss={avg_loss:.4f}  ppl={ppl:.2f}")
        except Exception as e:
            print(f"  {domain:12s}: [error: {e}]")

    results["domain_perplexity"] = domain_ppls

    # 2. Theology-specific generation samples
    print("\n--- Theology Generation Samples ---")
    theology_prompts = [
        "In the beginning",
        "And God said",
        "The Lord is my",
        "Blessed are the",
        "For God so loved",
    ]

    samples = generate_samples(
        model, tokenizer, theology_prompts, device,
        max_new_tokens=80, temperature=0.8, top_k=40,
    )
    results["theology_samples"] = samples

    for s in samples:
        print(f"\n  Prompt: \"{s['prompt']}\"")
        generated_part = s["generated"][len(s["prompt"]):].strip()
        # Wrap long lines
        display = generated_part[:200]
        print(f"  Output: {display}")

    # 3. General-domain generation (to see domain shift)
    print("\n--- General Generation Samples (testing domain bias) ---")
    general_prompts = [
        "The function returns",
        "Once upon a time",
        "The experiment shows",
    ]

    gen_samples = generate_samples(
        model, tokenizer, general_prompts, device,
        max_new_tokens=60, temperature=0.8, top_k=40,
    )
    results["general_samples"] = gen_samples

    for s in gen_samples:
        print(f"\n  Prompt: \"{s['prompt']}\"")
        generated_part = s["generated"][len(s["prompt"]):].strip()
        display = generated_part[:200]
        print(f"  Output: {display}")

    # 4. Summary statistics
    if "theology" in domain_ppls:
        theo_ppl = domain_ppls["theology"]["perplexity"]
        other_ppls = [v["perplexity"] for k, v in domain_ppls.items() if k != "theology"]
        avg_other = sum(other_ppls) / max(len(other_ppls), 1) if other_ppls else float("nan")

        print("\n--- Summary ---")
        print(f"  Theology perplexity:       {theo_ppl:.2f}")
        if other_ppls:
            print(f"  Avg other-domain ppl:      {avg_other:.2f}")
            print(f"  Domain specialization gap: {avg_other - theo_ppl:.2f}")
            results["specialization_gap"] = avg_other - theo_ppl

    results["model_params"] = count_parameters(model, trainable_only=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Theology-domain transformer pre-training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Tokenizer & model config
    tok = Tokenizer()
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=args.max_seq_len)
    print(f"Vocab size: {model_cfg.vocab_size} ({'HuggingFace' if tok._hf else 'byte-level'})")

    # Build transformer with LM head
    tcfg = TransformerConfig()
    expert = TransformerExpert(
        n_layers=tcfg.n_layers, d_model=tcfg.d_model, n_heads=tcfg.n_heads,
        head_dim=tcfg.head_dim, ffn_mult=tcfg.ffn_mult, dropout=tcfg.dropout,
        max_seq_len=model_cfg.max_seq_len, rope_base=tcfg.rope_base,
    )
    embedding = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
    model = ExpertWithHead(expert, embedding, tcfg.d_model, model_cfg.vocab_size).to(device)

    n_params = count_parameters(model, trainable_only=True)
    print(f"Trainable parameters: {n_params:,}")

    # Checkpoint dir
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "theology_transformer.pt")

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        print(f"  Loaded (epoch {state.get('epoch', '?')}, loss {state.get('loss', '?')})")

    amp_dtype = torch.bfloat16

    if not args.eval_only:
        # ---- TRAINING ----
        print(f"\n{'=' * 60}")
        print(f"Theology-domain Transformer Pre-training")
        print(f"{'=' * 60}")

        # Build theology-only dataloaders
        train_loader = build_dataloader(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            domains=["theology"],
            split="train",
            val_split=0.1,
            domain_balanced=False,
        )
        val_loader = build_dataloader(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            domains=["theology"],
            split="val",
            val_split=0.1,
            domain_balanced=False,
        )
        print(f"Theology train batches: {len(train_loader)} | val batches: {len(val_loader)}")
        print(f"Epochs: {args.max_epochs} | LR: {args.lr} | Batch size: {args.batch_size}")
        print(f"Seq len: {args.max_seq_len} | Warmup: {args.warmup_steps} steps")

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

        # LR schedule
        max_steps = len(train_loader) * args.max_epochs
        lr_fn = partial(get_lr, warmup_steps=args.warmup_steps, max_steps=max_steps, max_lr=args.lr)

        # AMP scaler (only for fp16 + CUDA)
        scaler = None

        best_val_loss = float("inf")
        patience_counter = 0
        step = 0
        training_log = []

        t_start = time.time()

        for epoch in range(args.max_epochs):
            train_loss, step = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler_fn=lr_fn,
                step=step,
                epoch=epoch,
                device=device,
                grad_clip=1.0,
                log_interval=50,
                scaler=scaler,
                amp_dtype=amp_dtype,
            )

            val_loss = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
            val_ppl = math.exp(min(val_loss, 20))

            epoch_log = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 2),
            }
            training_log.append(epoch_log)

            print(f"\nEpoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, step, epoch, val_loss, ckpt_path, scaler)
                print(f"  -> Best model saved to {ckpt_path}")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t_start
        print(f"\nTraining complete in {elapsed:.1f}s | Best val loss: {best_val_loss:.4f}")

        # Reload best checkpoint for evaluation
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            print(f"Loaded best checkpoint for evaluation")

    # ---- EVALUATION ----
    eval_results = run_evaluation(model, tok, device, amp_dtype, data_dir=args.data_dir)

    # Save results
    results_path = ckpt_dir / "theology_eval_results.json"
    serializable = {}
    for k, v in eval_results.items():
        if isinstance(v, (dict, list, int, float, str)):
            serializable[k] = v
    if not args.eval_only:
        serializable["training_log"] = training_log

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
