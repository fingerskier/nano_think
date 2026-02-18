# Training How-To

## Data File Format

Each data file is a JSONL file (one JSON object per line) with the following fields:

```json
{"text": "The actual text content...", "source": "wikitext", "id": 0}
```

| Field    | Required | Description                          |
|----------|----------|--------------------------------------|
| `text`   | Yes      | The text content for training        |
| `source` | No       | Source identifier (informational)    |
| `id`     | No       | Numeric identifier (informational)   |

Only the `text` field is read by the training pipeline. The `source` and `id` fields are metadata.

### Directory Layout

Data files live under `data/` in domain sub-directories:

```
data/
  academic/
    arxiv_abstracts.jsonl
    ml_arxiv.jsonl
  code/
    codeparrot.jsonl
  general/
    wikitext.jsonl
    tinystories.jsonl
  philosophy/
    stanford_plato.jsonl
  theology/
    openbible.jsonl
```

The directory name **is** the domain name. The data loader iterates over sub-directories of the data root, reads every `*.jsonl` file inside, and tags each sample with the domain from the directory name.

## Training on a Single File

Two approaches:

### Approach 1: Domain Filtering (recommended)

Place your file in a domain directory and use `--domains` to restrict training to that domain.

```bash
# 1. Put your file in an existing or new domain directory
mkdir -p data/my_domain
cp my_data.jsonl data/my_domain/

# 2. Train only on that domain
python scripts/pretrain_experts.py --expert transformer --domains my_domain
```

**Note:** The `--domains` flag is accepted by `build_dataloader()` via the `DataConfig.domains` list. If the training scripts don't expose `--domains` as a CLI flag directly, you can modify `DataConfig.domains` in `nano_think/config.py` to `["my_domain"]`.

### Approach 2: Isolated Data Directory

Create a directory containing only your target file and point `--data_dir` at the parent:

```bash
# 1. Set up an isolated directory structure
mkdir -p my_training_data/my_domain
cp my_data.jsonl my_training_data/my_domain/

# 2. Point --data_dir at the parent
python scripts/pretrain_experts.py --expert transformer --data_dir my_training_data
```

The file **must** be inside a sub-directory (the domain folder) — files placed directly in the data root are ignored.

## Training Phases

Training happens in two phases:

### Phase 1: Expert Pre-Training

Each expert (transformer, diffuser, state-space) is trained as a standalone language model with its own LM head.

```bash
# Train in order — diffuser depends on a pre-trained transformer encoder
python scripts/pretrain_experts.py --expert transformer
python scripts/pretrain_experts.py --expert state_space
python scripts/pretrain_experts.py --expert diffuser
```

- The **transformer** and **state_space** experts train with standard cross-entropy loss.
- The **diffuser** expert uses MSE diffusion loss against embeddings from a frozen pre-trained transformer. Train the transformer first.
- Checkpoints are saved to `checkpoints/<expert>_expert.pt`.

### Phase 2: MLA Training

The full NanoThink model (MLA attention + router) trains with expert weights **frozen**:

```bash
python scripts/train_mla.py
```

This loads the three expert checkpoints, freezes them, and trains the token embeddings, MLA attention layers, expert router, and output head.

Checkpoint is saved to `checkpoints/nanothink_mla.pt`.

## Resuming from a Checkpoint

If training is interrupted (crash, timeout, manual stop), you can resume from the last saved checkpoint using the `--resume` flag.

### Usage

```bash
# Auto-detect most recent checkpoint
python scripts/pretrain_experts.py --expert transformer --resume
python scripts/train_mla.py --resume

# Explicit checkpoint path
python scripts/pretrain_experts.py --expert transformer --resume checkpoints/transformer_expert.pt
python scripts/train_mla.py --resume checkpoints/nanothink_mla.pt
```

When `--resume` is used without a path, auto-detection searches `checkpoints/` in this priority order:

1. `*_interrupted.pt` — most likely what you want after Ctrl+C
2. Newest `*_epoch*.pt` — latest periodic snapshot
3. `*.pt` (best checkpoint) — may be older than the epoch snapshots

### What's Saved in a Checkpoint

| Field            | Description                                      |
|------------------|--------------------------------------------------|
| `model_state`    | Full model weights (`state_dict`)                |
| `optimizer_state` | Optimizer momentum / adaptive-rate buffers       |
| `step`           | Global training step (used for LR schedule)      |
| `epoch`          | Epoch number at save time                        |
| `amp_scaler`     | AMP `GradScaler` state (only when AMP is active) |

### What Happens on Resume

1. The checkpoint is loaded with `strict=False`, so minor architecture changes (added/removed parameters) won't cause a hard failure — mismatched keys are silently skipped.
2. The optimizer state is restored, preserving momentum and adaptive learning-rate buffers.
3. Training restarts at **epoch + 1** with the saved step counter, so the cosine LR schedule picks up where it left off.

### Checkpoint Locations

| File                                    | When Created                          |
|-----------------------------------------|---------------------------------------|
| `checkpoints/<expert>_expert.pt`        | Best validation loss during Phase 1   |
| `checkpoints/nanothink_mla.pt`         | Best validation loss during Phase 2   |
| `checkpoints/<expert>_expert_epoch{N}.pt` | Every 5 epochs (periodic snapshot) |
| `checkpoints/nanothink_mla_epoch{N}.pt`  | Every 5 epochs (periodic snapshot) |

To resume from a periodic snapshot instead of the best checkpoint, pass the epoch-specific path:

```bash
python scripts/pretrain_experts.py --expert transformer --resume checkpoints/transformer_expert_epoch10.pt
```

### Caveats

- **LR schedule recalculation:** The cosine schedule is recomputed from the restored `step` value. This works correctly for a straightforward resume. However, if you change `max_epochs` between runs, the cosine curve shifts — the schedule will still start from the correct step, but the total decay length changes.
- **`strict=False` loading:** Missing or extra keys in the checkpoint are silently ignored. This is intentional (it allows minor model changes between runs) but means a completely wrong checkpoint won't raise an error — it will just produce garbage. Double-check the checkpoint path matches the model you're training.

## CLI Reference

### `scripts/pretrain_experts.py`

| Flag         | Default  | Description                                    |
|--------------|----------|------------------------------------------------|
| `--expert`   | required | Expert to train: `transformer`, `diffuser`, `state_space` |
| `--data_dir` | `data`   | Root data directory                            |
| `--resume`   | `None`   | Resume from checkpoint (omit path to auto-detect) |
| `--device`   | auto     | Device (`cuda`, `cpu`, or specific like `cuda:0`) |

### `scripts/train_mla.py`

| Flag                 | Default  | Description                                   |
|----------------------|----------|-----------------------------------------------|
| `--data_dir`         | `data`   | Root data directory                           |
| `--resume`           | `None`   | Resume from checkpoint (omit path to auto-detect) |
| `--device`           | auto     | Device (`cuda`, `cpu`, or specific)           |
| `--use_vector_store` | `False`  | Enable FAISS vector store                     |

## Key Training Features

- **Automatic Mixed Precision (AMP):** Enabled by default. Uses `bf16` on supported hardware, falls back to `fp16`. Controlled by `TrainExpertConfig.use_amp` and `amp_dtype`.
- **Gradient Accumulation:** MLA training uses `grad_accum_steps=4` (effective batch size = `batch_size * 4 = 64`).
- **Gradient Clipping:** `grad_clip=1.0` by default.
- **Checkpointing:** Best model saved automatically (based on validation loss). Periodic checkpoints every 5 epochs.
- **Early Stopping:** Training stops after 5 epochs without validation loss improvement.
- **Domain-Balanced Sampling:** `WeightedRandomSampler` gives equal weight to each domain during training. Disabled for validation.
- **Learning Rate Schedule:** Cosine decay with linear warmup (`warmup_steps` configurable).
- **Router Entropy Regularisation (MLA):** Penalises low router entropy to encourage expert diversity (`entropy_reg=0.01`).
- **Deterministic Splits:** Train/val split uses a fixed seed (42) for reproducibility.

## Configuration

All hyperparameters are defined as dataclasses in `nano_think/config.py`:

| Config Class         | Purpose                            | Key Defaults                       |
|----------------------|------------------------------------|------------------------------------|
| `ModelConfig`        | Shared dimensions                  | `d_model=384`, `max_seq_len=512`   |
| `TransformerConfig`  | Transformer expert                 | `n_layers=6`, `ffn_mult=4`         |
| `DiffuserConfig`     | Diffuser expert                    | `n_layers=4`, `training_steps=100` |
| `StateSpaceConfig`   | State-space (S6) expert            | `n_layers=6`, `d_state=16`         |
| `MLAConfig`          | MLA attention + router             | `n_experts=3`, `n_heads=6`         |
| `TrainExpertConfig`  | Expert training hyperparams        | `lr=3e-4`, `batch_size=32`, `max_epochs=20` |
| `TrainDiffuserConfig`| Diffuser-specific overrides        | `lr=1e-4`, `batch_size=64`         |
| `TrainMLAConfig`     | MLA training hyperparams           | `lr=1e-4`, `batch_size=16`, `max_epochs=30` |
| `DataConfig`         | Data pipeline                      | `tokenizer=gpt2`, `val_split=0.1`  |

To customise, either modify the dataclass defaults or (where supported) pass CLI arguments.
