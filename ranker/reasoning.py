"""Stage 6 — feature-driven reasoning generator.

The reasoning column is human-graded (constraint.md): it must cite *specific* facts (years, title,
named corroborated skills, signal values), connect to a JD requirement, raise a concern *only when a
real gap exists*, vary between rows, never hallucinate, and tone-match the rank. Every clause here is
conditioned on a computed feature, so the text can only ever state what the data supports.

Variation and tone-matching fall out of the data: different candidates trip different clauses, and the
confidence word + whether concerns surface is banded by rank. The concern clauses are pulled from the
SAME signals that drove the score (penalty factors, availability notes, band edges), so the explanation
can never contradict the ranking.
"""

from __future__ import annotations

from . import config


def _category_clauses(text: str) -> list[str]:
    """Concrete JD-core evidence found in the career text (retrieval / vector / ranking)."""
    has = lambda kws: any(k in text for k in kws)
    retrieval = has(config.KW_RETRIEVAL)
    vector = has(config.KW_VECTOR_DB)
    ranking = has(config.KW_RANKING_EVAL)
    clauses = []
    if retrieval and ranking:
        clauses.append("career evidence of building both retrieval and ranking systems")
    elif retrieval:
        clauses.append("hands-on embeddings/retrieval work in role history")
    elif ranking:
        clauses.append("ranking/recommendation-systems experience")
    if vector:
        clauses.append("vector-search / hybrid-retrieval infrastructure")
    return clauses


def _concerns(f: dict, comp: dict, trust: float, avail_notes: list[str]) -> list[str]:
    """Real gaps only — each tied to the signal that actually moved the score."""
    out: list[str] = []
    pen = comp["penalties"]
    yoe = f["years_of_experience"]
    if pen["consulting"] < 1.0:
        out.append("services-firm-heavy background")
    if pen["noncoding"] < 1.0:
        out.append("recent title leans toward architecture/management, not hands-on code")
    if pen["domain"] < 1.0:
        out.append("core domain is CV/speech/robotics with limited NLP/IR")
    if pen["stability"] < 1.0:
        out.append("short average tenure — possible title-chasing")
    if yoe < config.EXP_BAND[0]:
        out.append(f"{yoe:.0f}y is just below the 5-9y band")
    elif yoe > config.EXP_BAND[1]:
        out.append(f"{yoe:.0f}y sits above the typical band")
    if comp["location"] <= 0.3:
        out.append("based outside India with relocation unclear (no visa sponsorship)")
    if trust < 0.25:
        out.append("self-reported skills lightly corroborated by career evidence")
    out.extend(avail_notes)
    return out


def generate_reasoning(
    f: dict,
    comp: dict,
    trust: float,
    corroborated: list[str],
    avail_notes: list[str],
    rank: int,
) -> str:
    """One to two sentences, grounded and rank-toned."""
    title = f["current_title"] or "Candidate"
    yoe = f["years_of_experience"]
    text = f"{f['career_text']} {f['summary']} {f['headline']}".lower()

    # Confidence word, banded by rank (tone-match-rank).
    lead_word = "Excellent fit" if rank <= 10 else "Strong fit" if rank <= 40 else "Solid fit"

    # Order by specificity: the core category clause, then the candidate's OWN corroborated skills
    # (the most varied, fact-rich clause), then broader signals. Named skills give each row a
    # distinct fingerprint and satisfy the "cite specific facts / vary between rows" grading bar.
    cats = _category_clauses(text)
    strengths: list[str] = cats[:1]
    if trust >= 0.55 and corroborated:
        named = ", ".join(corroborated[:3])
        strengths.append(f"skills corroborated in role history ({named})")
    if comp["pre_llm"] >= 1.0:
        strengths.append("pre-2022 ML foundation, not just recent LLM tooling")
    strengths.extend(cats[1:])  # vector-search infra etc., if room remains
    if comp["nice_to_have"] >= 0.6:
        strengths.append("nice-to-haves present (LoRA/LTR/OSS or distributed-systems exposure)")
    if not strengths and comp["semantic"] >= 0.55:
        strengths.append("profile aligns semantically with the JD's retrieval/ranking mandate")

    # JD connection — pick the most relevant hook (varies by the strongest strength).
    if cats and "both retrieval and ranking" in cats[0]:
        hook = "maps directly to owning Redrob's ranking & retrieval layer"
    elif any("rank" in s for s in strengths):
        hook = "fits the evaluation-framework bar (NDCG/MRR/MAP, offline-to-online)"
    else:
        hook = "aligns with the production-ML-systems mandate"

    head = f"{lead_word}: {title}, {yoe:.0f}y"
    body = "; ".join(strengths[:3]) if strengths else "general ML engineering background"
    sentence = f"{head} — {body}; {hook}."

    # Concerns: top ranks only surface a concern if one is material; lower ranks always note the top gap.
    concerns = _concerns(f, comp, trust, avail_notes)
    if concerns:
        keep = concerns[: (1 if rank <= 10 else 2)]
        sentence += " Concern: " + "; ".join(keep) + "."
    return sentence
