"""FAISS-backed vector store for retrieval-augmented memory."""

from __future__ import annotations

import numpy as np
import torch

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    """Dense retrieval index over 384D vectors using FAISS (inner product on L2-normed vecs).

    Provides a persistent, growing memory that augments the model's context window.
    """

    def __init__(self, d_model: int = 384, use_gpu: bool = False):
        if faiss is None:
            raise ImportError("faiss is required: pip install faiss-cpu (or faiss-gpu)")

        self.d_model = d_model
        self.use_gpu = use_gpu

        # Inner product on L2-normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(d_model)

        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, vectors: torch.Tensor) -> None:
        """L2-normalize and insert vectors into the index.

        Args:
            vectors: (N, d_model) tensor
        """
        vecs = vectors.detach().float().cpu().numpy()
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms
        self.index.add(vecs)

    def search(self, query: torch.Tensor, k: int = 4) -> torch.Tensor:
        """Retrieve top-k nearest 384D vectors for each query.

        Args:
            query: (B, d_model) tensor
            k: number of neighbors

        Returns:
            (B, k, d_model) tensor of retrieved vectors
        """
        if self.index.ntotal == 0:
            # Return zeros if store is empty
            B = query.shape[0]
            return torch.zeros(B, k, self.d_model, device=query.device, dtype=query.dtype)

        actual_k = min(k, self.index.ntotal)

        q = query.detach().float().cpu().numpy()
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        q = q / norms

        _, indices = self.index.search(q, actual_k)  # (B, k)

        # Reconstruct vectors from indices
        results = []
        for i in range(q.shape[0]):
            row_vecs = []
            for idx in indices[i]:
                if idx == -1:
                    row_vecs.append(np.zeros(self.d_model, dtype=np.float32))
                else:
                    vec = np.zeros(self.d_model, dtype=np.float32)
                    self.index.reconstruct(int(idx), vec)
                    row_vecs.append(vec)
            # Pad if we got fewer than k results
            while len(row_vecs) < k:
                row_vecs.append(np.zeros(self.d_model, dtype=np.float32))
            results.append(np.stack(row_vecs))

        result = np.stack(results)  # (B, k, d_model)
        return torch.from_numpy(result).to(device=query.device, dtype=query.dtype)

    def reset(self) -> None:
        """Clear all vectors from the index."""
        self.index.reset()
