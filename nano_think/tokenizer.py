"""Tokenizer for nano_think.

Supports two backends:
  1. HuggingFace transformers (default, e.g. "gpt2")
  2. Byte-level (explicit opt-in via name="bytes", no network required)
"""

from __future__ import annotations


# ---- Byte-level fallback tokenizer (offline) --------------------------------

_PAD_ID = 0
_EOS_ID = 1
_BOS_ID = 2
_BYTE_OFFSET = 3  # byte values 0-255 are mapped to IDs 3-258
_BYTE_VOCAB_SIZE = 259  # 256 bytes + 3 special tokens


class _ByteTokenizer:
    """Minimal byte-level tokenizer — maps raw UTF-8 bytes to token IDs."""

    vocab_size: int = _BYTE_VOCAB_SIZE
    pad_id: int = _PAD_ID
    eos_id: int = _EOS_ID
    bos_id: int = _BOS_ID

    def encode(self, text: str) -> list[int]:
        return [b + _BYTE_OFFSET for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        byte_vals = []
        for i in ids:
            if i < _BYTE_OFFSET:
                continue  # skip special tokens
            byte_vals.append(i - _BYTE_OFFSET)
        return bytes(byte_vals).decode("utf-8", errors="replace")


# ---- Public wrapper ----------------------------------------------------------


class Tokenizer:
    """Tokenizer wrapper: HuggingFace by default, byte-level via name='bytes'."""

    def __init__(self, name: str = "gpt2", max_len: int = 512):
        self.max_len = max_len
        self._hf = None
        self._byte = None

        if name == "bytes":
            self._byte = _ByteTokenizer()
            return

        from transformers import AutoTokenizer

        # Try cache first (no network, no warning)
        try:
            self._hf = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:
            # Not cached yet — download once
            self._hf = AutoTokenizer.from_pretrained(name)

        if self._hf.pad_token is None:
            self._hf.pad_token = self._hf.eos_token

    # ---- properties ----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        if self._hf is not None:
            return len(self._hf)
        return self._byte.vocab_size

    @property
    def pad_id(self) -> int:
        if self._hf is not None:
            return self._hf.pad_token_id
        return _PAD_ID

    @property
    def eos_id(self) -> int:
        if self._hf is not None:
            return self._hf.eos_token_id
        return _EOS_ID

    @property
    def bos_id(self) -> int:
        if self._hf is not None:
            return self._hf.bos_token_id or self._hf.eos_token_id
        return _BOS_ID

    # ---- encode / decode -----------------------------------------------------

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        """Encode text to token IDs."""
        if self._hf is not None:
            ids = self._hf.encode(text)
        else:
            ids = self._byte.encode(text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self._hf is not None:
            return self._hf.decode(ids, skip_special_tokens=True)
        return self._byte.decode(ids)

    def batch_encode(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        """Batch-encode with padding / truncation (HuggingFace only, fallback raises)."""
        if self._hf is not None:
            return self._hf(
                texts,
                padding=padding,
                truncation=truncation,
                max_length=self.max_len,
                return_tensors="pt",
            )
        raise NotImplementedError("batch_encode is only supported with a HuggingFace tokenizer")
