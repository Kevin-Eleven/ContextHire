"""Skill-trust correction (anti-stuffing).
This is the one stage that *reads* skills[] — but only as corroboration, never as evidence of fit
(the JD's stated trap). The principle that ties the whole system together: a self-reported skill is
trustworthy only to the degree the candidate's actual career corroborates it. The decisive signal is
whether the skill name appears in the career text — a keyword-stuffer's "embeddings / RAG / ranking"
claims never surface in their marketing/HR job descriptions, so they earn ~no trust; a genuine ML
engineer's relevant skills are all over their role descriptions.

CALIBRATION — measured by calibrate.py over all 960,302 skills in the full 100K pool (run
`.venv/bin/python calibrate.py`, see calibration_report.txt): endorsements med 8 / p90 15 ·
duration med 16mo / p90 33 · assessment present for only 3.7% of skills (rare → strong when high) ·
proficiency "expert" is 0.14% of skills (notable). So career-text presence is weighted highest, with
assessment/endorsements/duration as secondary boosts.

Returns a trust score in [0,1] (a positive pillar in the blend, config.WEIGHTS["skill_trust"]) plus the
names of the best-corroborated JD-relevant skills, which Stage 6 reasoning cites by name.
"""

from __future__ import annotations

from . import config

# Skills that matter for THIS role; an endorsed "Java" tells us nothing about fit here.
_RELEVANT = (
    config.KW_RETRIEVAL
    | config.KW_VECTOR_DB
    | config.KW_RANKING_EVAL
    | config.KW_ML_GENERAL
)
# ~2.5 fully-corroborated relevant skills saturates trust at 1.0.
_SATURATION = 2.5


def _is_relevant(name: str) -> bool:
    n = name.lower()
    return any(kw in n or n in kw for kw in _RELEVANT)


def _corroboration(skill: dict, career_text: str) -> float:
    """How much real evidence backs this one claimed skill, in [0, 1]."""
    c = 0.0
    name = skill["name"].lower()
    if name and name in career_text:  # the linchpin: skill shows up in actual work
        c += 0.6
    a = skill["assessment"]
    if a is not None:
        c += 0.3 if a >= 55 else (0.15 if a >= 40 else 0.0)
    if skill["endorsements"] >= 12:  # near p90=15 (calibrate.py) — top-decile peer endorsement
        c += 0.15
    if skill["duration_months"] >= 24:  # well above med=16mo, approaching p90=33 (calibrate.py)
        c += 0.10
    return min(1.0, c)


def skill_trust(f: dict) -> tuple[float, list[str]]:
    """Return (trust in [0,1], best-corroborated relevant skill names, strongest first)."""
    career_text = f["career_text"].lower()
    scored = [
        (s["name"], _corroboration(s, career_text))
        for s in f["skills"]
        if _is_relevant(s["name"])
    ]
    if not scored:
        return 0.0, []
    trust = min(1.0, sum(c for _, c in scored) / _SATURATION)
    corroborated = [name for name, c in sorted(scored, key=lambda x: -x[1]) if c >= 0.6]
    return trust, corroborated
