"""Cosine similarity of a query vector against a whole index, on the GPU when possible."""
import numpy as np

from vibekimus import settings
from vibekimus.identity_cache import IdentityCache
from vibekimus.runlog import safe_print

BYTES_PER_HALF_FLOAT = 2
# The GPU copy of an index may take at most this share of the free VRAM.
MAX_SHARE_OF_FREE_VRAM = 0.5
# Rows moved to the GPU / multiplied per step, to keep transient memory small.
UPLOAD_ROWS_PER_STEP = 131_072
MULTIPLY_ROWS_PER_STEP = 262_144


def top_indices(scores, min_score, limit):
    """Indices of the best `limit` scores that are >= min_score (unordered)."""
    indices = np.flatnonzero(scores >= min_score)
    if indices.size > limit:
        best = np.argpartition(scores[indices], indices.size - limit)[-limit:]
        indices = indices[best]
    return indices


class SimilarityEngine:
    """Scores every index row against one query vector.

    With a CUDA device each index is kept there in half precision (about 2.3 GB per
    million rows of 1152 floats) and a search over millions of rows takes
    milliseconds. The copy is rebuilt whenever the index array is replaced, and
    everything falls back to numpy on the CPU if the GPU is unavailable or too full.
    """

    def __init__(self):
        self._gpu_copies = IdentityCache()   # index name -> GPU tensor of that numpy array

    def invalidate(self):
        self._gpu_copies.clear()

    def scores(self, name, embeddings, vector, device=None):
        """embeddings: (N, D) float32; vector: D floats. Returns (N,) float32 numpy."""
        vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if settings.GPU_SEARCH and device is not None and embeddings.shape[0] > 0:
            try:
                gpu_copy = self._gpu_copy(name, embeddings, device)
                if gpu_copy is not None:
                    return self._multiply_on_gpu(gpu_copy, vector, device)
            except Exception as e:
                safe_print(f"[SEARCH] GPU search unavailable ({e}), using CPU")
                self._gpu_copies.drop(name)
        return embeddings @ vector

    def _gpu_copy(self, name, embeddings, device):
        """The index as a half-precision GPU tensor, or None if it does not fit."""
        import torch
        cached = self._gpu_copies.get(name, embeddings)
        if cached is not None:
            return cached
        self._gpu_copies.drop(name)   # the copy of a replaced array frees its graphics memory first
        free_bytes, _total = torch.cuda.mem_get_info()
        if embeddings.size * BYTES_PER_HALF_FLOAT >= free_bytes * MAX_SHARE_OF_FREE_VRAM:
            return None
        gpu_copy = torch.empty(embeddings.shape, dtype=torch.float16, device=device)
        for start in range(0, embeddings.shape[0], UPLOAD_ROWS_PER_STEP):
            chunk = embeddings[start:start + UPLOAD_ROWS_PER_STEP]
            gpu_copy[start:start + UPLOAD_ROWS_PER_STEP] = torch.from_numpy(chunk).to(device)
        return self._gpu_copies.put(name, gpu_copy, embeddings)

    @staticmethod
    def _multiply_on_gpu(gpu_copy, vector, device):
        import torch
        query = torch.from_numpy(vector).to(device)
        # Multiply in float32 chunk by chunk — exact scores, small transient memory.
        parts = [gpu_copy[start:start + MULTIPLY_ROWS_PER_STEP].float() @ query
                 for start in range(0, gpu_copy.shape[0], MULTIPLY_ROWS_PER_STEP)]
        return torch.cat(parts).cpu().numpy()
