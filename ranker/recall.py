"""Semantic recall index (ranking-time, numpy-only).
Loads the offline artifacts (precompute_embeddings.py) and exposes one semantic fit score per
candidate in [0, 1], fusing dense and lexical signals:
  - dense:  cosine(JD, candidate) from the bge-small embeddings, computed here as a single
            [N,384]·[384] matmul (<0.1s for 100K), then min-max normalized across the pool.
  - lexical: precomputed, already-normalized BM25(JD, candidate).
Fusion is a fixed convex blend (dense-leaning): dense carries the semantic understanding the JD demands; BM25 anchors it to the concrete must-have vocabulary so a purely "vibey" summary can't float up without the real retrieval/ranking terms in the career text.
"""

from __future__ import annotations

import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3


class SemanticIndex:
    def __init__(self, data_dir: str = _DATA):
        self.available = False
        try:
            emb = np.load(os.path.join(data_dir, "embeddings.npy")).astype(np.float32)
            ids = np.load(os.path.join(data_dir, "candidate_ids.npy"))
            jd = np.load(os.path.join(data_dir, "jd_embedding.npy")).astype(np.float32)
            bm25 = np.load(os.path.join(data_dir, "bm25.npy")).astype(np.float32)
        except FileNotFoundError:
            return

        # Dense cosine for the whole pool in one matmul (vectors are L2-normalized at precompute).
        dense = emb @ jd
        lo, hi = float(dense.min()), float(dense.max())
        dense_norm = (dense - lo) / (hi - lo) if hi > lo else np.zeros_like(dense)

        fused = DENSE_WEIGHT * dense_norm + BM25_WEIGHT * bm25
        self._score = {cid: float(s) for cid, s in zip(ids.tolist(), fused)}
        self.available = True

    def semantic_score(self, candidate_id: str) -> float:
        """Fused dense+lexical fit in [0, 1]; 0.0 for an unknown id (no embedding precomputed)."""
        return self._score.get(candidate_id, 0.0)
