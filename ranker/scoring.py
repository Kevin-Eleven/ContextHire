"""Stage B — JD rubric scorer.
Operationalizes the fixed JD (Senior AI Engineer — Founding Team, Redrob AI) into a numeric
fit score in (0, 1]. The governing principle, stated explicitly by the JD itself: the `skills[]`
array is an adversarial trap, so fit is judged from *career evidence* — job titles and the
free-text role descriptions/summary — never from the self-reported skill list. (Skill-list
corroboration is handled separately as a trust signal in Stage C.)

Design: a weighted sum of positive components (what the JD wants) multiplied by bounded penalty
factors (what the JD explicitly does NOT want). `score_rubric` returns both the scalar and a
component breakdown so Stage 6 can ground its reasoning and we can debug rankings.
"""

from __future__ import annotations

from . import config

# --- Title tiers: how well a job title alone fits an applied-AI-engineering role --------
# The decisive anti-honeypot signal — a "Marketing Manager" with a perfect AI skill list lands
# in the bottom tier no matter what its skills say.

_TITLE_TIERS = (
    (
        1.00,
        (
            "ai engineer",
            "ml engineer",
            "machine learning",
            "data scientist",
            "applied scientist",
            "nlp",
            "deep learning",
            "mlops",
            "ai/ml",
            "ml scientist",
            "research engineer",
            "ai researcher",
            "ai/ ml",
        ),
    ),
    (
        0.60,
        ("data engineer", "analytics engineer", "data analyst", "research scientist"),
    ),
    (
        0.50,
        (
            "software engineer",
            "backend",
            "back end",
            "platform engineer",
            "search engineer",
            "full-stack",
            "cloud engineer",
            "devops",
        ),
    ),
    (
        0.30,
        (
            "frontend",
            "front end",
            "mobile developer",
            "qa engineer",
            "java developer",
            ".net developer",
            "developer",
        ),
    ),
)
_NONFIT_TIER = 0.05  # everything else (marketing/HR/sales/accountant/civil/etc.)

# Component weights for the positive blend (sum to 1.0). Role + evidence dominate because they
# define fit and the scoring rewards the top-10 most (NDCG@10 = half the composite).

RUBRIC_WEIGHTS = {
    "role": 0.30,
    "evidence": 0.30,
    "experience": 0.15,
    "pre_llm": 0.10,
    "location": 0.10,
    "nice_to_have": 0.05,
}


def _has_any(text: str, kws) -> bool:
    return any(k in text for k in kws)


def _title_tier(title: str) -> float:
    t = title.lower()
    for base, kws in _TITLE_TIERS:
        if any(k in t for k in kws):
            return base
    return _NONFIT_TIER


def _role_fit(f: dict) -> float:
    """Best title tier across the career — current title at full weight, history at 0.9."""
    cur = _title_tier(f["current_title"])
    hist = max((_title_tier(c["title"]) for c in f["careers"]), default=0.0)
    return max(cur, 0.9 * hist)


def _evidence(text: str) -> float:
    """Evidence of the JD's 'absolutely need' work, found in career text (not skills[])."""
    core_cats = (config.KW_RETRIEVAL, config.KW_VECTOR_DB, config.KW_RANKING_EVAL)
    core = sum(1 for kws in core_cats if _has_any(text, kws)) / len(core_cats)
    ml_general = 1.0 if _has_any(text, config.KW_ML_GENERAL) else 0.0
    return 0.7 * core + 0.3 * ml_general


def _experience(yoe: float) -> float:
    lo, hi = config.EXP_IDEAL  # 6-8 ideal -> 1.0
    blo, bhi = config.EXP_BAND  # 5-9 band  -> still strong
    if lo <= yoe <= hi:
        return 1.0
    if blo <= yoe <= bhi:
        return 0.85
    gap = (blo - yoe) if yoe < blo else (yoe - bhi)
    return max(0.0, 0.85 - gap / 8.0)  # juniors and the very senior taper off


def _pre_llm(f: dict, text: str) -> float:
    """JD wants people who 'understood retrieval and ranking before it became fashionable' and
    explicitly down-weights 'AI = only recent LangChain'. Reward pre-2022 ML evidence.
    """
    ml_titleish = any(_title_tier(c["title"]) >= 0.6 for c in f["careers"])
    for c in f["careers"]:
        if c["start_date"] and c["start_date"].year <= 2021 and ml_titleish:
            return 1.0
    # Some ML signal but all recent: partial credit scaled by total experience.
    if _has_any(text, config.KW_ML_GENERAL) or ml_titleish:
        return 0.5 if f["years_of_experience"] >= 4 else 0.25
    return 0.2


def _location(f: dict) -> float:
    loc = f["location"].lower()
    if any(city in loc for city in config.TARGET_CITIES):
        return 1.0
    if "india" in f["country"].lower():
        return 0.7  # in India, can relocate to Pune/Noida
    if f["behavioral"]["willing_to_relocate"]:
        return 0.6
    return 0.2  # abroad, won't relocate (JD: no visa sponsorship)


def _nice_to_have(f: dict, text: str) -> float:
    score = 0.0
    if _has_any(text, config.KW_NICE_TO_HAVE):
        score += 0.6
    gh = f["behavioral"]["github_activity_score"]
    if gh is not None and gh >= 40:  # OSS / external validation signal
        score += 0.4
    return min(1.0, score)


# --- bounded penalty factors (things the JD explicitly does NOT want) -------------------
def _consulting_factor(f: dict) -> float:
    companies = [c["company"].lower() for c in f["careers"] if c["company"]]
    if not companies:
        return 1.0
    matches = [any(firm in co for firm in config.CONSULTING_FIRMS) for co in companies]
    if all(matches):
        return 0.55  # entire career in services firms
    cur = f["current_company"].lower()
    if any(firm in cur for firm in config.CONSULTING_FIRMS):
        return 0.85  # currently at a services firm but has other history
    return 1.0


def _noncoding_factor(f: dict) -> float:
    # JD: rejects seniors who stopped writing code (architect/manager/director).
    return (
        0.7 if _has_any(f["current_title"].lower(), config.NON_CODING_TITLES) else 1.0
    )


def _domain_factor(f: dict, text: str) -> float:
    # JD: CV/speech/robotics primary without NLP/IR exposure is a poor fit.
    if _has_any(text, config.KW_NON_FIT_DOMAIN) and not _has_any(
        text, config.KW_RETRIEVAL | {"nlp", "information retrieval", "ranking"}
    ):
        return 0.6
    return 1.0


def _stability_factor(f: dict) -> float:
    # JD: title-chasers who switch every ~1.5 years are not a fit.
    if f["n_roles"] >= 4 and 0 < f["avg_tenure_months"] < 18:
        return 0.75
    return 1.0


def score_rubric(f: dict) -> tuple[float, dict]:
    """Return (rubric_score in (0,1], component breakdown)."""
    text = f"{f['career_text']} {f['summary']} {f['headline']}".lower()

    comp = {
        "role": _role_fit(f),
        "evidence": _evidence(text),
        "experience": _experience(f["years_of_experience"]),
        "pre_llm": _pre_llm(f, text),
        "location": _location(f),
        "nice_to_have": _nice_to_have(f, text),
    }
    positive = sum(RUBRIC_WEIGHTS[k] * v for k, v in comp.items())

    penalties = {
        "consulting": _consulting_factor(f),
        "noncoding": _noncoding_factor(f),
        "domain": _domain_factor(f, text),
        "stability": _stability_factor(f),
    }
    factor = 1.0
    for v in penalties.values():
        factor *= v

    comp["penalties"] = penalties
    score = round(max(1e-6, positive * factor), 6)
    return score, comp
