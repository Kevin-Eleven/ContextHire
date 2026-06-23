#!/usr/bin/env python3
"""One-shot dataset calibration — the source of every measured constant in the ranker.

Run ONCE, offline, over the full candidates.jsonl. Every "CALIBRATION:" comment in
ranker/behavioral.py, ranker/skill_trust.py and ranker/honeypot.py quotes a number this
script prints, so the thresholds and cliffs in those files sit on real distribution points
instead of guesses. Re-run it if the dataset ever changes; the comments should match its output.

    .venv/bin/python calibrate.py [--candidates ./candidates.jsonl] [--out calibration_report.txt]

It streams the pool line-by-line (O(1) in the file size beyond the per-signal value arrays,
which are ~100K floats each — a few MB) and needs only numpy + the ranker package.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time

import numpy as np

from ranker import config
from ranker.features import _parse_date, extract
from ranker.honeypot import honeypot_reasons
from ranker.io import iter_candidates


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _fmt_numeric(name: str, vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if not a.size:
        return f"  {name:<28} (no data)"
    return (
        f"  {name:<28} min {a.min():.4g} · p10 {_pct(a,10):.4g} · med {_pct(a,50):.4g} · "
        f"p90 {_pct(a,90):.4g} · max {a.max():.4g}  (n={a.size})"
    )


def calibrate(path: str) -> list[str]:
    """Single streaming pass; returns the report as a list of lines."""
    # --- behavioral signal value arrays (redrob_signals) ---
    sig_vals: dict[str, list[float]] = {
        k: []
        for k in (
            "profile_completeness",
            "recruiter_response_rate",
            "last_active_days",
            "interview_completion_rate",
            "notice_period_days",
            "saved_by_recruiters_30d",
        )
    }
    n_total = 0
    n_open_to_work = 0
    n_willing_relocate = 0

    # --- skill corroboration fields ---
    skill_endorsements: list[float] = []
    skill_durations: list[float] = []
    n_skills = 0
    n_skill_assessment = 0  # skills that have an assessment score
    n_skill_expert = 0  # proficiency == "expert"

    # --- honeypot band diagnostics ---
    n_salary_inverted = 0
    n_salary_present = 0
    n_last_before_signup = 0
    n_signup_present = 0
    duration_span_delta: list[float] = []  # duration_months - dated span (per role)
    yoe_span_delta: list[float] = []  # yoe - documented span (untruncated histories)
    n_flagged = 0

    t0 = time.time()
    for raw in iter_candidates(path):
        n_total += 1
        f = extract(raw)
        b = f["behavioral"]

        sig_vals["profile_completeness"].append(b["profile_completeness"])
        sig_vals["recruiter_response_rate"].append(b["recruiter_response_rate"])
        if b["last_active_days"] is not None:
            sig_vals["last_active_days"].append(b["last_active_days"])
        sig_vals["interview_completion_rate"].append(b["interview_completion_rate"])
        sig_vals["notice_period_days"].append(b["notice_period_days"])
        sig_vals["saved_by_recruiters_30d"].append(b["saved_by_recruiters_30d"])
        n_open_to_work += bool(b["open_to_work"])
        n_willing_relocate += bool(b["willing_to_relocate"])

        for s in f["skills"]:
            n_skills += 1
            skill_endorsements.append(s["endorsements"])
            skill_durations.append(s["duration_months"])
            if s["assessment"] is not None:
                n_skill_assessment += 1
            if (s["proficiency"] or "").lower() == "expert":
                n_skill_expert += 1

        # honeypot diagnostics straight off the raw signal block
        rs = raw.get("redrob_signals", {}) or {}
        salary = rs.get("expected_salary_range_inr_lpa", {}) or {}
        smin, smax = salary.get("min"), salary.get("max")
        if smin is not None and smax is not None:
            n_salary_present += 1
            if smin > smax:
                n_salary_inverted += 1
        signup = _parse_date(rs.get("signup_date"))
        last_active = _parse_date(rs.get("last_active_date"))
        if signup and last_active:
            n_signup_present += 1
            if last_active < signup:
                n_last_before_signup += 1

        for c in f["careers"]:
            start = c["start_date"]
            end = config.REFERENCE_DATE if c["is_current"] else c["end_date"]
            if start and end and c["duration_months"] > 0:
                span = (end - start).days / 30.44
                duration_span_delta.append(c["duration_months"] - span)
        starts = [c["start_date"].year for c in f["careers"] if c["start_date"]]
        if starts and f["n_roles"] < 10:
            documented = config.REFERENCE_DATE.year - min(starts)
            yoe_span_delta.append(f["years_of_experience"] - documented)

        if honeypot_reasons(f):
            n_flagged += 1

        if n_total % 20000 == 0:
            print(f"  ...{n_total} scanned ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ---- assemble report ----
    def share(n: int, d: int) -> str:
        return f"{(100.0*n/d if d else 0.0):.2f}%"

    L: list[str] = []
    L.append(f"DATASET CALIBRATION — {n_total} candidates, scanned in {time.time()-t0:.0f}s")
    L.append("")
    L.append("[ranker/behavioral.py]  redrob_signals availability distributions")
    for name, vals in sig_vals.items():
        L.append(_fmt_numeric(name, vals))
    L.append(f"  {'open_to_work == true':<28} {share(n_open_to_work, n_total)}")
    L.append(f"  {'willing_to_relocate == true':<28} {share(n_willing_relocate, n_total)}")
    L.append("")

    L.append(f"[ranker/skill_trust.py]  skill corroboration fields  (n_skills={n_skills})")
    L.append(_fmt_numeric("endorsements", skill_endorsements))
    L.append(_fmt_numeric("duration_months", skill_durations))
    L.append(f"  {'has assessment score':<28} {share(n_skill_assessment, n_skills)} of skills")
    L.append(f"  {'proficiency == expert':<28} {share(n_skill_expert, n_skills)} of skills")
    L.append("")

    L.append("[ranker/honeypot.py]  impossibility-signal diagnostics")
    L.append(
        f"  {'salary min > max':<28} {share(n_salary_inverted, n_salary_present)} "
        f"(of {n_salary_present} with a range) — data convention, NOT a honeypot"
    )
    L.append(
        f"  {'last_active < signup':<28} {share(n_last_before_signup, n_signup_present)} "
        f"(of {n_signup_present} with both dates)"
    )
    dsd = np.asarray(duration_span_delta)
    if dsd.size:
        within = 100.0 * np.mean(np.abs(dsd) <= 0.8)
        L.append(
            f"  {'role duration vs dated span':<28} {within:.2f}% within 0.8mo · "
            f"max overclaim {dsd.max():.0f}mo  (n_roles={dsd.size})"
        )
    ysd = np.asarray(yoe_span_delta)
    if ysd.size:
        within = 100.0 * np.mean(np.abs(ysd) <= 0.8)
        L.append(
            f"  {'yoe vs documented span':<28} {within:.2f}% within 0.8y · "
            f"max overclaim {ysd.max():.0f}y  (untruncated, n={ysd.size})"
        )
    L.append(
        f"  {'honeypot filter flags':<28} {n_flagged}/{n_total} "
        f"({share(n_flagged, n_total)})"
    )
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute the dataset constants the ranker cites.")
    ap.add_argument("--candidates", default=config.DEFAULT_CANDIDATES)
    ap.add_argument("--out", default="calibration_report.txt")
    args = ap.parse_args()

    lines = calibrate(args.candidates)
    report = "\n".join(lines)
    print("\n" + report + "\n")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(f"# Generated by calibrate.py on {_dt.date.today()}\n")
        fh.write(report + "\n")
    print(f"(written to {args.out})")


if __name__ == "__main__":
    main()
