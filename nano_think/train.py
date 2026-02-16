"""Training utilities — loops, schedulers, logging."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """Cosine decay learning rate with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loss: float,
    path: str,
    scaler: GradScaler | None = None,
) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> dict:
    """Load training checkpoint. Returns metadata dict."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return {"step": state.get("step", 0), "epoch": state.get("epoch", 0), "loss": state.get("loss", float("inf"))}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler_fn,
    step: int,
    epoch: int,
    device: torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 50,
    scaler: GradScaler | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    grad_accum_steps: int = 1,
    compute_loss_fn=None,
) -> tuple[float, int]:
    """Run one training epoch.

    Args:
        compute_loss_fn: Optional function (model, batch, device, amp_dtype) -> (loss, extra_dict).
                         If None, uses default cross-entropy on model output logits.

    Returns:
        (average_loss, updated_step)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Update LR
        lr = scheduler_fn(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler is not None):
            if compute_loss_fn is not None:
                loss, extra = compute_loss_fn(model, batch, device, amp_dtype)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad()
            step += 1

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        if batch_idx % log_interval == 0:
            avg = total_loss / n_batches
            ppl = math.exp(min(avg, 20))
            print(
                f"  epoch {epoch} | step {step} | batch {batch_idx}/{len(dataloader)} | "
                f"loss {avg:.4f} | ppl {ppl:.2f} | lr {lr:.2e}"
            )

    return total_loss / max(n_batches, 1), step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    compute_loss_fn=None,
) -> float:
    """Evaluate model on a DataLoader, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
            if compute_loss_fn is not None:
                loss, _ = compute_loss_fn(model, batch, device, amp_dtype)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
