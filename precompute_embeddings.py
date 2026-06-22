#!/usr/bin/env python3
"""OFFLINE precompute (declared in submission_metadata.yaml: pre_computation_required: true).

NOT part of the 5-minute ranking budget and the ONLY step allowed to use the network / a model.
It encodes every candidate's *evidence text* (headline + summary + career titles & descriptions —
never skills[], per the JD's stated trap) with BAAI/bge-small-en-v1.5, encodes the fixed JD as a
query, and precomputes BM25 lexical scores against that JD. Everything is written to data/ as
compact .npy so rank.py needs only numpy at ranking time — no torch, no model, no network.

Artifacts (data/):
  embeddings.npy     float16 [N,384]  L2-normalized candidate vectors (dot == cosine)
  candidate_ids.npy  <U16   [N]       row-aligned candidate ids
  jd_embedding.npy   float32 [384]    the JD query vector (same space)
  bm25.npy           float16 [N]      min-max-normalized BM25(JD, candidate) in [0,1]

Run:  .venv/bin/python precompute_embeddings.py
"""

from __future__ import annotations

import os
import re
import time

import numpy as np

from ranker.features import extract
from ranker.io import iter_candidates
from ranker.scoring import score_rubric

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE is asymmetric: the query (only) gets this instruction prefix; passages get none.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OUT_DIR = "data"
BATCH = 128
MAX_SEQ = 192  # career descriptions can be long; the JD-relevant signal is in the lead.
# Only the strongest rubric contenders are embedded — semantic just reorders the top, and the
# rest can't reach the top 100 anyway. Generous margin so no real contender is excluded.
SHORTLIST = 12000

# A dense, natural-language restatement of what the JD "absolutely needs" + strong positives.
# This is the semantic query the whole pool is matched against (skills[] deliberately excluded).
JD_QUERY = (
    "Senior AI / machine learning engineer who builds production embeddings-based retrieval and "
    "semantic search systems, hybrid search and vector databases (FAISS, Pinecone, Qdrant, "
    "Elasticsearch, OpenSearch), and ranking / recommendation systems. Hands-on with designing "
    "evaluation frameworks for ranking quality — NDCG, MRR, MAP, offline-to-online correlation, "
    "A/B testing. Strong Python. Deep pre-LLM machine learning and information retrieval / NLP "
    "experience, learning-to-rank, LLM fine-tuning (LoRA, QLoRA, PEFT). Ships fast at an "
    "early-stage product company; owns the ranking and matching intelligence layer end to end."
)
# Lexical query terms for BM25 (the concrete must-have vocabulary).
BM25_QUERY = (
    "embeddings retrieval semantic search vector database hybrid search faiss pinecone qdrant "
    "elasticsearch opensearch ranking recommendation recsys learning to rank ndcg mrr map "
    "evaluation ab testing python nlp information retrieval fine-tuning lora machine learning"
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # --- pass 1: stream the pool once; collect ids, evidence text, cheap rubric fit ----------
    ids: list[str] = []
    texts: list[str] = []
    rubric: list[float] = []
    for raw in iter_candidates("candidates.jsonl"):
        f = extract(raw)
        ids.append(f["candidate_id"])
        # Same evidence the rubric reads — keeps dense/lexical/rubric signals consistent.
        texts.append(f["evidence_text"] or f["current_title"] or "")
        rubric.append(score_rubric(f)[0])
    print(f"[{time.time() - t0:.0f}s] loaded {len(ids)} candidates")

    # Keep only the top-SHORTLIST by rubric fit; embed just those (huge encode-time saving).
    order = np.argsort(rubric)[::-1][:SHORTLIST]
    ids = [ids[i] for i in order]
    texts = [texts[i] for i in order]
    n = len(ids)
    print(f"[{time.time() - t0:.0f}s] shortlisted {n} contenders for embedding")

    # --- dense embeddings (the expensive part) -----------------------------------------------
    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_SEQ
    emb = model.encode(
        texts,
        batch_size=BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float16)
    jd_vec = model.encode(
        BGE_QUERY_PREFIX + JD_QUERY, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    print(f"[{time.time() - t0:.0f}s] encoded dense {emb.shape}")

    # --- BM25 lexical scores against the JD --------------------------------------------------
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    raw_scores = np.asarray(bm25.get_scores(_tokenize(BM25_QUERY)), dtype=np.float32)
    lo, hi = float(raw_scores.min()), float(raw_scores.max())
    bm25_norm = ((raw_scores - lo) / (hi - lo)) if hi > lo else np.zeros(n, np.float32)
    print(f"[{time.time() - t0:.0f}s] BM25 scored (raw max {hi:.2f})")

    # --- persist -----------------------------------------------------------------------------
    np.save(f"{OUT_DIR}/embeddings.npy", emb)
    np.save(f"{OUT_DIR}/candidate_ids.npy", np.asarray(ids, dtype="<U16"))
    np.save(f"{OUT_DIR}/jd_embedding.npy", jd_vec)
    np.save(f"{OUT_DIR}/bm25.npy", bm25_norm.astype(np.float16))

    mb = sum(
        os.path.getsize(f"{OUT_DIR}/{x}")
        for x in ("embeddings.npy", "candidate_ids.npy", "jd_embedding.npy", "bm25.npy")
    ) / 1e6
    print(f"[{time.time() - t0:.0f}s] wrote {OUT_DIR}/ artifacts — {mb:.1f} MB total")


if __name__ == "__main__":
    main()
