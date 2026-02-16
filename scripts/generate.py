#!/usr/bin/env python3
"""Autoregressive text generation with a trained NanoThink model.

Usage:
    python scripts/generate.py --checkpoint checkpoints/nanothink_mla.pt --prompt "The meaning of life"
    python scripts/generate.py --checkpoint checkpoints/nanothink_mla.pt --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nano_think.config import (
    DiffuserConfig,
    MLAConfig,
    ModelConfig,
    StateSpaceConfig,
    TransformerConfig,
    VectorStoreConfig,
)
from nano_think.model import NanoThink
from nano_think.tokenizer import Tokenizer


def load_model(checkpoint_path: str, device: torch.device, vocab_size: int = 259) -> NanoThink:
    """Load a trained NanoThink model from checkpoint."""
    model_cfg = ModelConfig(vocab_size=vocab_size)
    model = NanoThink(
        model_cfg=model_cfg,
        transformer_cfg=TransformerConfig(),
        diffuser_cfg=DiffuserConfig(),
        ssm_cfg=StateSpaceConfig(),
        mla_cfg=MLAConfig(),
        vs_cfg=VectorStoreConfig(),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {state.get('epoch', '?')} | Step: {state.get('step', '?')} | Loss: {state.get('loss', '?'):.4f}")
    return model


def generate_text(
    model: NanoThink,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
    show_routing: bool = False,
) -> str:
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    generated = tokenizer.decode(output_ids[0].tolist())

    if show_routing:
        # Run one forward pass to get routing weights
        with torch.no_grad():
            outputs = model(input_tensor)
            weights = outputs["router_weights"][0].cpu().tolist()
            print(f"  Router weights: Transformer={weights[0]:.3f} | Diffuser={weights[1]:.3f} | SSM={weights[2]:.3f}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text with NanoThink")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--show_routing", action="store_true", help="Display expert routing weights")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = Tokenizer()
    model = load_model(args.checkpoint, device, vocab_size=tokenizer.vocab_size)

    if args.interactive:
        print("\nNanoThink Interactive Generation")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break

            if prompt.strip().lower() in ("quit", "exit"):
                break
            if not prompt.strip():
                continue

            text = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
                show_routing=args.show_routing,
            )
            print(f"\n{text}\n")

    elif args.prompt:
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            show_routing=args.show_routing,
        )
        print(text)

    else:
        parser.print_help()
        print("\nProvide --prompt or --interactive")


if __name__ == "__main__":
    main()
