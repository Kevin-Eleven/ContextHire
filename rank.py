#!/usr/bin/env python3
"""Entry point — produces the top-100 submission CSV from candidates.jsonl.
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

This is the command declared as `reproduce_command` and must run within the compute budget
(CPU-only, no network, <=5 min, <=16GB; see constraint.md).

STAGE 1 STATUS: end-to-end plumbing only. Selection/format logic is final, but the score is a
transparent PLACEHOLDER (experience-band proxy). Stages 2-6 replace `score_candidate` /
`build_reasoning` with the honeypot filter, JD rubric, behavioral modifier, and real reasoning.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import time

from ranker import config
from ranker.features import extract
from ranker.honeypot import is_honeypot
from ranker.io import iter_candidates
from ranker.scoring import score_rubric


def build_reasoning(f: dict, comp: dict) -> str:
    """Interim reasoning grounded in the rubric components — replaced by Stage 6 templating."""
    title = f["current_title"] or "Unknown role"
    bits = [f"{title}, {f['years_of_experience']:.1f}y exp"]
    if comp["evidence"] >= 0.4:
        bits.append("career shows retrieval/ranking/ML work")
    if comp["role"] < 0.3:
        bits.append("title is a poor fit for the role")
    if comp["penalties"]["consulting"] < 1.0:
        bits.append("services-firm background")
    return "; ".join(bits) + "."


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
    for raw in iter_candidates(path):
        f = extract(raw)
        # Stage 0: drop internally-impossible profiles before they can reach the top 100.
        if is_honeypot(f):
            n_honeypots += 1
            continue
        score, comp = score_rubric(f)
        rec = {
            "candidate_id": f["candidate_id"],
            "score": score,
            "reasoning": build_reasoning(f, comp),
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
