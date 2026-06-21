"""Structured feature extraction from a raw candidate profile.
This stage only *reads* the profile into a normalized, score-ready dict. Keeping extraction declarative makes the sentinel handling (github=-1, offer_acceptance=-1, end_date=null) auditable in one place.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from .config import REFERENCE_DATE


def _parse_date(s: Any) -> _dt.date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return _dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _days_since(d: _dt.date | None) -> float | None:
    if d is None:
        return None
    return (REFERENCE_DATE - d).days


def _text(*parts: str) -> str:
    return " ".join(p for p in parts if p).strip()


def extract(raw: dict) -> dict:
    """Normalize one candidate dict into a flat feature dict used by every later stage."""
    prof = raw.get("profile", {}) or {}
    career = raw.get("career_history", []) or []
    edu = raw.get("education", []) or []
    skills = raw.get("skills", []) or []
    certifications = raw.get("certifications", []) or []
    languages = raw.get("languages", {}) or {}
    sig = raw.get("redrob_signals", {}) or {}

    # --- career history ---
    careers = []
    for job in career:
        careers.append(
            {
                "company": (job.get("company") or "").strip(),
                "title": (job.get("title") or "").strip(),
                "duration_months": int(job.get("duration_months") or 0),
                "is_current": bool(job.get("is_current")),
                "industry": (job.get("industry") or "").strip(),
                "company_size": job.get("company_size") or "",
                "start_date": _parse_date(job.get("start_date")),
                "end_date_raw": job.get("end_date"),
                "end_date": _parse_date(job.get("end_date")),
                "description": (job.get("description") or "").strip(),
            }
        )
    durations = [c["duration_months"] for c in careers if c["duration_months"] > 0]
    avg_tenure_months = (sum(durations) / len(durations)) if durations else 0.0

    # Recency of most recent role end (None end_date == current role == 0 days).
    end_recency_days = None
    if careers:
        ends = [0 if c["is_current"] else _days_since(c["end_date"]) for c in careers]
        ends = [e for e in ends if e is not None]
        if ends:
            end_recency_days = min(ends)

    # --- evidence text: summary + career titles/descriptions (NOT skills[]) ---
    career_text = " ".join(_text(c["title"], c["description"]) for c in careers)

    evidence_text = _text(
        prof.get("headline", ""), prof.get("summary", ""), career_text
    )

    # --- skills with corroboration fields ---
    assess = sig.get("skill_assessment_scores", {}) or {}
    # Also using the redrob signals skill assesment scores
    skill_list = []
    for s in skills:
        name = (s.get("name") or "").strip()
        skill_list.append(
            {
                "name": name,
                "proficiency": s.get("proficiency", ""),
                "endorsements": int(s.get("endorsements") or 0),
                "duration_months": int(s.get("duration_months") or 0),
                "assessment": float(assess[name]) if name in assess else None,
            }
        )

    # --- behavioral signals (sentinels preserved as None where meaningful) ---
    gh = sig.get("github_activity_score")
    oar = sig.get("offer_acceptance_rate")
    salary = sig.get("expected_salary_range_inr_lpa", {}) or {}
    behavioral = {
        "profile_completeness": float(sig.get("profile_completeness_score") or 0.0),
        "open_to_work": bool(sig.get("open_to_work_flag")),
        "recruiter_response_rate": float(sig.get("recruiter_response_rate") or 0.0),
        "avg_response_time_hours": float(sig.get("avg_response_time_hours") or 0.0),
        "last_active_days": _days_since(_parse_date(sig.get("last_active_date"))),
        "interview_completion_rate": float(sig.get("interview_completion_rate") or 0.0),
        "offer_acceptance_rate": None if oar is None or oar < 0 else float(oar),
        "saved_by_recruiters_30d": int(sig.get("saved_by_recruiters_30d") or 0),
        "github_activity_score": None if gh is None or gh < 0 else float(gh),
        "notice_period_days": int(sig.get("notice_period_days") or 0),
        "willing_to_relocate": bool(sig.get("willing_to_relocate")),
        "preferred_work_mode": sig.get("preferred_work_mode", ""),
        "expected_salary_min": salary.get("min"),
        "expected_salary_max": salary.get("max"),
    }

    return {
        "candidate_id": raw.get("candidate_id", ""),
        "current_title": (prof.get("current_title") or "").strip(),
        "headline": (prof.get("headline") or "").strip(),
        "summary": (prof.get("summary") or "").strip(),
        "location": (prof.get("location") or "").strip(),
        "country": (prof.get("country") or "").strip(),
        "years_of_experience": float(prof.get("years_of_experience") or 0.0),
        "current_company": (prof.get("current_company") or "").strip(),
        "current_industry": (prof.get("current_industry") or "").strip(),
        "careers": careers,
        "n_roles": len(careers),
        "avg_tenure_months": avg_tenure_months,
        "end_recency_days": end_recency_days,
        "education": edu,
        "skills": skill_list,
        "career_text": career_text,
        "evidence_text": evidence_text,
        "behavioral": behavioral,
    }
