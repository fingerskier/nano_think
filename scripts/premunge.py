#!/usr/bin/env python3
"""Pre-chunk training data into max_seq_len-sized windows.

Reads JSONL from data/{domain}/*.jsonl, tokenizes each text, splits into
sliding windows of (max_seq_len - 1) tokens (reserving 1 for EOS at training
time), decodes back to text, and writes to data_prepared/{domain}/*.jsonl.

Usage:
    python scripts/premunge.py
    python scripts/premunge.py --data_dir data --output_dir data_prepared
    python scripts/premunge.py --stride 256 --min_chunk_tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nano_think.tokenizer import Tokenizer


def chunk_text_by_tokens(
    text: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    stride: int,
    min_chunk_tokens: int,
) -> list[dict]:
    """Tokenize text and split into chunks of max_seq_len-1 tokens.

    Returns list of dicts with keys: text, token_count, chunk_idx, total_chunks.
    """
    # Encode without EOS — training adds it
    ids = tokenizer.encode(text, add_eos=False)

    chunk_size = max_seq_len - 1  # reserve 1 for EOS appended at training time

    if len(ids) <= chunk_size:
        return [{"text": text, "token_count": len(ids), "chunk_idx": 0, "total_chunks": 1}]

    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        chunk_ids = ids[start:end]

        # Discard short tail fragments from multi-chunk splits
        if len(chunk_ids) < min_chunk_tokens and len(chunks) > 0:
            break

        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_ids),
            "chunk_idx": len(chunks),
        })

        start += stride

    # Backfill total_chunks
    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total

    return chunks


def process_file(
    jsonl_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    max_seq_len: int,
    stride: int,
    min_chunk_tokens: int,
) -> dict:
    """Process a single JSONL file. Returns stats dict."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_count = 0
    chunk_count = 0
    skipped_short = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            text = entry.get("text", "")
            if not text.strip():
                continue

            original_count += 1
            original_id = entry.get("id", str(original_count))

            chunks = chunk_text_by_tokens(text, tokenizer, max_seq_len, stride, min_chunk_tokens)

            for chunk in chunks:
                record = {
                    "text": chunk["text"],
                    "source": entry.get("source", jsonl_path.stem),
                    "id": f"{original_id}_chunk{chunk['chunk_idx']}",
                    "chunk_idx": chunk["chunk_idx"],
                    "total_chunks": chunk["total_chunks"],
                    "original_id": original_id,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

    return {
        "file": jsonl_path.name,
        "original_samples": original_count,
        "chunks": chunk_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-chunk training data into fixed-size token windows")
    parser.add_argument("--data_dir", type=str, default="data", help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="data_prepared", help="Output directory")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length (tokens)")
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding window (default: max_seq_len-1, no overlap)")
    parser.add_argument("--min_chunk_tokens", type=int, default=64, help="Minimum tokens per chunk (discard shorter tails)")
    parser.add_argument("--domains", nargs="*", default=None, help="Domains to process (default: all)")
    args = parser.parse_args()

    if args.stride is None:
        args.stride = args.max_seq_len - 1  # no overlap by default

    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)

    if not data_path.exists():
        print(f"Error: data directory '{data_path}' does not exist")
        sys.exit(1)

    print(f"Pre-munging training data")
    print(f"  Input:       {data_path}")
    print(f"  Output:      {output_path}")
    print(f"  Tokenizer:   {args.tokenizer_name}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Stride:      {args.stride}")
    print(f"  Min chunk:   {args.min_chunk_tokens} tokens")
    print()

    tokenizer = Tokenizer(args.tokenizer_name, max_len=args.max_seq_len)

    all_stats = {}
    t0 = time.time()

    for domain_dir in sorted(data_path.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        if args.domains and domain not in args.domains:
            continue

        domain_stats = {"original_samples": 0, "chunks": 0, "files": []}

        for jsonl_file in sorted(domain_dir.glob("*.jsonl")):
            out_file = output_path / domain / jsonl_file.name
            stats = process_file(
                jsonl_file, out_file, tokenizer,
                args.max_seq_len, args.stride, args.min_chunk_tokens,
            )
            domain_stats["files"].append(stats)
            domain_stats["original_samples"] += stats["original_samples"]
            domain_stats["chunks"] += stats["chunks"]

        all_stats[domain] = domain_stats
        print(f"  {domain:20s}  {domain_stats['original_samples']:>6d} samples -> {domain_stats['chunks']:>6d} chunks")

    elapsed = time.time() - t0

    # Write metadata
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "tokenizer_name": args.tokenizer_name,
        "max_seq_len": args.max_seq_len,
        "stride": args.stride,
        "min_chunk_tokens": args.min_chunk_tokens,
        "domains": {k: {"original_samples": v["original_samples"], "chunks": v["chunks"]} for k, v in all_stats.items()},
        "elapsed_seconds": round(elapsed, 2),
    }
    meta_path = output_path / "_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    total_orig = sum(s["original_samples"] for s in all_stats.values())
    total_chunks = sum(s["chunks"] for s in all_stats.values())
    print(f"\n  {'TOTAL':20s}  {total_orig:>6d} samples -> {total_chunks:>6d} chunks")
    print(f"\nDone in {elapsed:.1f}s. Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
