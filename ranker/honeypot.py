"""Honeypot Impossibility Filter
The dataset contains ~80 honeypots: "subtly impossible profiles. It targets *internal logical impossibility* (a contradiction a real profile cannot have), which is distinct from keyword-stuffing / role mismatch (handled by scoring.py).
CALIBRATION — every figure below is measured by calibrate.py over the full 100K pool (run
`.venv/bin/python calibrate.py`, see calibration_report.txt). Most "impossibility-looking" fields
are just synthetic-data conventions and fire on 8-20% of candidates — useless as honeypot signals:
  - expected_salary min > max          → 18.86% of candidates (a data convention, NOT a honeypot)
  - last_active_date < signup_date     → 7.50%
  - skill duration_months > yoe*12      → smooth noise (skill durations are independently sampled)
  - skill anachronism                  → dead (skill durations are capped at 96mo, per calibrate.py)
Two signals, by contrast, are razor-clean — ~99.9% of the pool sits in a tight band and a tiny
tail jumps far past it, exactly the planted impossibilities:
  - duration_months vs the role's start->end span: 99.88% match within 0.8mo; tail reaches 189mo.
  - years_of_experience vs documented career span: 99.90% within 0.8y; tail reaches ~13y.
Those two (plus the trivially-true is_current/end_date contradiction) catch 44 profiles with no
false positives. We deliberately under-claim rather than risk evicting a genuine candidate.
"""

from __future__ import annotations

import datetime as _dt

from .config import REFERENCE_DATE

# A role's stated duration may exceed its date span by at most this (rounding slack).
DURATION_SPAN_TOLERANCE_MONTHS = 6.0
# Claimed years_of_experience may exceed the documented (non-truncated) span by at most this.
YOE_SPAN_TOLERANCE_YEARS = 6.0
# career_history is capped at 10 roles by the schema; at the cap we assume truncation and skip
# the yoe-vs-span check so we never penalize a genuinely deep history.
CAREER_HISTORY_MAXITEMS = 10


def _months_between(a: _dt.date, b: _dt.date) -> float:
    return (b - a).days / 30.44


def honeypot_reasons(f: dict) -> list[str]:
    """Return the list of impossibility reasons; empty list == not a honeypot."""
    reasons: list[str] = []

    # 1. Per-role: stated duration contradicts the start/end dates (fabricated tenure), or a role
    #    that ends before it starts. For a current role the span runs to the reference date.
    for c in f["careers"]:
        if c["is_current"] and c["end_date_raw"] not in (None, ""):
            reasons.append(
                f"is_current but end_date={c['end_date_raw']} ({c['title']})"
            )
        start = c["start_date"]
        end = REFERENCE_DATE if c["is_current"] else c["end_date"]
        if start and end and c["duration_months"] > 0:
            span = _months_between(start, end)
            if span < -1:
                reasons.append(f"role ends before it starts ({c['title']})")
            elif c["duration_months"] - span > DURATION_SPAN_TOLERANCE_MONTHS:
                reasons.append(
                    f"duration {c['duration_months']}mo but dates span only ~{span:.0f}mo "
                    f"({c['title']})"
                )

    # 2. Claimed experience far exceeds the documented career span — only when career_history is
    #    NOT truncated (so deep, truncated histories are never falsely flagged).
    starts = [c["start_date"].year for c in f["careers"] if c["start_date"]]
    if starts and f["n_roles"] < CAREER_HISTORY_MAXITEMS:
        documented_span = REFERENCE_DATE.year - min(starts)
        if f["years_of_experience"] - documented_span > YOE_SPAN_TOLERANCE_YEARS:
            reasons.append(
                f"claims {f['years_of_experience']:.0f}y experience but career is "
                f"documented over only ~{documented_span}y"
            )

    return reasons


def is_honeypot(f: dict) -> bool:
    return bool(honeypot_reasons(f))


def _scan(path: str, examples: int = 15) -> None:
    """Standalone scan: report how many candidates the filter flags, with a few examples."""
    from .features import extract
    from .io import iter_candidates

    total = flagged = shown = 0
    for raw in iter_candidates(path):
        total += 1
        rs = honeypot_reasons(extract(raw))
        if rs:
            flagged += 1
            if shown < examples:
                print(f"  {raw.get('candidate_id')}: {rs[0]}")
                shown += 1
    pct = 100.0 * flagged / total if total else 0.0
    print(f"\nFlagged {flagged}/{total} ({pct:.3f}%) as impossible/honeypot.")


if __name__ == "__main__":
    import sys
    from .config import DEFAULT_CANDIDATES

    _scan(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CANDIDATES)
