"""
The `redrob_signals` block describes *availability / responsiveness*, not fit. How reachable, active, and hireable a candidate is right now. The plan is explicit
that this is a **bounded multiplier applied on top of the rubric**, never a weighted addend — it
breaks ties among similar-fit candidates and down-weights the unavailable, but must never let availability outrank genuine fit. So we model it as 1.0 minus *named* availability deductions,
clamped to a floor (config.BEHAVIORAL_MODIFIER_FLOOR = 0.5): a strong, reachable candidate keeps
their full rubric score (×1.0, no reshuffling of the fit elite); only real concerns pull it down.

CALIBRATION — every range/median below is measured by calibrate.py over the full 100K pool, so the cliffs sit
at real distribution points, not guesses:
  profile_completeness 25-99.9 (med 56.8) · recruiter_response_rate 0.02-0.95 (med 0.44)
  last_active_days 25-265 (med 130) · interview_completion_rate 0.30-1.00 (med 0.62)
  notice_period_days 0-150 (med 90, p90 150) · saved_by_recruiters_30d 0-80 (med 7, p90 15)
  open_to_work true for 35.3% · willing_to_relocate true for 28.8%

Each deduction returns a human-readable note.
"""

from __future__ import annotations

from .config import BEHAVIORAL_MODIFIER_FLOOR


def behavioral_modifier(f: dict) -> tuple[float, list[str]]:
    """Return (multiplier in [FLOOR, 1.0], list of availability concern notes)."""
    b = f["behavioral"]
    deductions = 0.0
    notes: list[str] = []

    # Not actively job-seeking — the single strongest availability signal (only 35.3% are open;
    # see calibrate.py).
    if not b["open_to_work"]:
        deductions += 0.10
        notes.append("not flagged open to work")

    # Profile dormancy: no penalty up to ~90d, ramping to a 0.12 cap by ~240d inactive.
    d = b["last_active_days"]
    if d is not None and d > 90:
        deductions += min(0.12, 0.12 * (d - 90) / (240 - 90))
        if d > 180:
            notes.append(f"inactive ~{int(d)}d")

    # Recruiter responsiveness: penalize only the genuinely unresponsive tail (rate < 0.30).
    rr = b["recruiter_response_rate"]
    if rr < 0.30:
        deductions += min(0.08, 0.08 * (0.30 - rr) / 0.30)
        if rr < 0.20:
            notes.append(f"low recruiter response rate ({rr:.0%})")

    # Long notice period — a real cost for a founding-team hire (med 90d; penalize the >=90 tail).
    np_ = b["notice_period_days"]
    if np_ >= 90:
        deductions += min(0.08, 0.08 * (np_ - 90) / (150 - 90))
        if np_ >= 120:
            notes.append(f"{np_}-day notice period")

    # Drops out of interview processes.
    if b["interview_completion_rate"] < 0.45:
        deductions += 0.06
        notes.append("low interview completion")

    # Thin/abandoned profile.
    if b["profile_completeness"] < 40:
        deductions += 0.04

    # Demand signal: heavily saved by recruiters → small positive nudge (market corroboration).
    # Threshold 15 == the p90 of saved_by_recruiters_30d (calibrate.py): only the top decile.
    bonus = 0.03 if b["saved_by_recruiters_30d"] >= 15 else 0.0

    mod = max(BEHAVIORAL_MODIFIER_FLOOR, min(1.0, 1.0 - deductions + bonus))
    return mod, notes
