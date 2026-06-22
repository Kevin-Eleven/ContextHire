#!/usr/bin/env python3
"""Entry point — produces the top-100 submission CSV from candidates.jsonl.
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

This is the command declared as `reproduce_command` and must run within the compute budget
(CPU-only, no network, <=5 min, <=16GB; see constraint.md).

Cascade: Stage 0 honeypot filter -> positive blend of Stage A semantic recall, Stage B JD rubric,
Stage C skill-trust -> Stage B penalty factors -> Stage D behavioral multiplier. The top 100 then
get Stage 6 feature-driven reasoning (deferred until ranks are known so tone can match rank).
"""

from __future__ import annotations

import argparse
import csv
import heapq
import time

from ranker import config
from ranker.behavioral import behavioral_modifier
from ranker.features import extract
from ranker.honeypot import is_honeypot
from ranker.io import iter_candidates
from ranker.reasoning import generate_reasoning
from ranker.recall import SemanticIndex
from ranker.scoring import score_rubric
from ranker.skill_trust import skill_trust


def _blend_weights(have_semantic: bool) -> tuple[float, float, float]:
    """(semantic, rubric, skill_trust) shares summing to 1.0; drop+renormalize if no embeddings."""
    w = config.WEIGHTS
    sem, rub, trust = w["semantic"], w["rubric"], w["skill_trust"]
    if not have_semantic:
        sem = 0.0
    total = sem + rub + trust
    return sem / total, rub / total, trust / total


def _id_num(cid: str) -> int:
    """Numeric part of CAND_XXXXXXX for deterministic tie-breaking."""
    try:
        return int(cid.split("_")[1])
    except (IndexError, ValueError):
        return 10**9


def select_top(path: str, top_n: int) -> list[dict]:
    """Stream the pool and keep the best `top_n` in a bounded min-heap (O(top_n) memory)."""
    # Heap key = (score, -id_num): the smallest element is the "worst" (lowest score, or on a
    # score tie the larger candidate_id), so popping evicts exactly who we'd rank last.
    heap: list[tuple] = []
    counter = 0  # unique, keeps heapq from ever comparing the dict payloads
    n_honeypots = 0
    index = SemanticIndex()  # Stage A: precomputed dense+BM25 fit (numpy-only, may be absent)
    print("Semantic index:", "loaded" if index.available else "absent (rubric-only fallback)")
    w_sem, w_rub, w_trust = _blend_weights(index.available)
    for raw in iter_candidates(path):
        f = extract(raw)
        # Stage 0: drop internally-impossible profiles before they can reach the top 100.
        if is_honeypot(f):
            n_honeypots += 1
            continue
        _, comp = score_rubric(f)
        # Stage A + C: blend semantic and skill-trust into the positive side, before penalties.
        semantic = index.semantic_score(f["candidate_id"]) if index.available else 0.0
        trust, corroborated = skill_trust(f)
        comp["semantic"] = semantic
        positive = w_sem * semantic + w_rub * comp["positive"] + w_trust * trust
        fit = positive * comp["factor"]
        # Stage D: bounded availability multiplier on top of fit (never dominates; floor 0.5).
        mod, avail_notes = behavioral_modifier(f)
        score = round(fit * mod, 6)
        # Carry what Stage 6 reasoning needs; generated after ranks are known (tone-match-rank).
        rec = {
            "candidate_id": f["candidate_id"],
            "score": score,
            "f": f,
            "comp": comp,
            "trust": trust,
            "corroborated": corroborated,
            "avail_notes": avail_notes,
        }
        key = (score, -_id_num(f["candidate_id"]))
        if len(heap) < top_n:
            heapq.heappush(heap, (key, counter, rec))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, counter, rec))
        counter += 1

    # Final order: best first — score desc, then candidate_id asc (validator tie-break rule).
    ranked = [item[2] for item in heap]
    ranked.sort(key=lambda r: (-r["score"], _id_num(r["candidate_id"])))
    # Stage 6: reasoning now that each candidate's final rank is fixed.
    for rank, rec in enumerate(ranked, start=1):
        rec["reasoning"] = generate_reasoning(
            rec["f"], rec["comp"], rec["trust"], rec["corroborated"], rec["avail_notes"], rank
        )
    print(f"Excluded {n_honeypots} honeypot/impossible profiles from contention.")
    return ranked


def write_submission(ranked: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, rec in enumerate(ranked, start=1):
            w.writerow(
                [rec["candidate_id"], i, f"{rec['score']:.6f}", rec["reasoning"]]
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Senior AI Engineer JD."
    )
    ap.add_argument(
        "--candidates",
        default=config.DEFAULT_CANDIDATES,
        help="Path to candidates.jsonl(.gz)",
    )
    ap.add_argument("--out", default=config.DEFAULT_OUTPUT, help="Output CSV path")
    ap.add_argument("--top-n", type=int, default=config.TOP_N)
    args = ap.parse_args()

    t0 = time.time()
    ranked = select_top(args.candidates, args.top_n)
    if len(ranked) < args.top_n:
        raise SystemExit(f"Only {len(ranked)} candidates found; need {args.top_n}.")
    write_submission(ranked, args.out)
    print(
        f"Wrote {len(ranked)} ranked candidates to {args.out} in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
