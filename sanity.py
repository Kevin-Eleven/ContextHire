"""Local sanity harness
Runs the real ranking pipeline and eyeballs the result and prints the top-30 with their reasoning, then runs automated red-flag checks.
"""

from __future__ import annotations

import argparse
import sys

from ranker import config
from ranker.honeypot import honeypot_reasons
from ranker.scoring import _title_tier
from rank import select_top

_NONFIT_TITLE = 0.05


def _best_title_tier(f: dict) -> float:
    tiers = [_title_tier(f["current_title"])]
    tiers += [_title_tier(c["title"]) for c in f["careers"]]
    return max(tiers, default=0.0)


def run(path: str, top_n: int) -> int:
    ranked = select_top(path, top_n)
    print(f"\n=== TOP {min(30, len(ranked))} (of {len(ranked)} ranked) ===\n")
    for i, rec in enumerate(ranked[:30], start=1):
        f = rec["f"]
        print(
            f"{i:>3}. {rec['score']:.4f}  {rec['candidate_id']}  "
            f"{f['current_title']!r} · {f['years_of_experience']:.0f}y · {f['location']}"
        )
        print(f"      {rec['reasoning']}")

    # ---- hard checks (failing any of these should block a submission) ----
    failures: list[str] = []

    honeypots = [r["candidate_id"] for r in ranked if honeypot_reasons(r["f"])]
    if honeypots:
        failures.append(
            f"{len(honeypots)} honeypot(s) in the ranked set: {honeypots[:5]}"
        )

    nonfit = [
        (r["candidate_id"], r["f"]["current_title"])
        for r in ranked
        if _best_title_tier(r["f"]) <= _NONFIT_TITLE
    ]
    if nonfit:
        failures.append(
            f"{len(nonfit)} non-fit title(s) in the ranked set: {nonfit[:5]}"
        )

    scores = [r["score"] for r in ranked]
    if scores != sorted(scores, reverse=True):
        failures.append("scores are not non-increasing by rank (validator will reject)")
    if len(set(scores)) < max(1, len(scores) // 2):
        failures.append(
            f"score gradient is flat ({len(set(scores))} distinct of {len(scores)}) — weak NDCG"
        )

    reasonings = [r["reasoning"] for r in ranked]
    dup = len(reasonings) - len(set(reasonings))
    if dup > len(reasonings) * 0.1:
        failures.append(
            f"{dup} duplicate reasoning strings (>10%) — grader penalizes repetition"
        )

    # ---- soft stats (informational) ----
    n_concern = sum(1 for r in reasonings if "Concern:" in r)
    print("\n=== SANITY ===")
    print(
        f"  ranked={len(ranked)}  distinct_scores={len(set(scores))}  "
        f"score_range={min(scores):.4f}-{max(scores):.4f}"
    )
    print(
        f"  distinct_reasonings={len(set(reasonings))}/{len(reasonings)}  with_concern={n_concern}"
    )
    print(f"  honeypots_in_top={len(honeypots)}  nonfit_titles_in_top={len(nonfit)}")

    if failures:
        print("\n❌ HARD CHECKS FAILED:")
        for msg in failures:
            print(f"   - {msg}")
        return 1
    print("\n✅ All hard checks passed — safe to validate & submit.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eyeball + gate the ranking before submitting."
    )
    ap.add_argument("--candidates", default=config.DEFAULT_CANDIDATES)
    ap.add_argument("--top-n", type=int, default=config.TOP_N)
    args = ap.parse_args()
    sys.exit(run(args.candidates, args.top_n))


if __name__ == "__main__":
    main()
