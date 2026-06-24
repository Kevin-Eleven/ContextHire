#!/usr/bin/env python3
"""Streamlit sandbox
Runs the ranking pipeline (the same `select_top` that produces the submission CSV) on a small candidate sample, and shows the ranked table + reasoning with a CSV
download. It deliberately reuses the production code path rather than a reimplementation.
Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import streamlit as st

from rank import select_top, write_submission

REPO = Path(__file__).resolve().parent
SAMPLE = REPO / "sample_candidates.json"

st.set_page_config(page_title="ContextHire | Candidate Ranking", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 2.5rem; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ContextHire")
st.caption(
    "Candidate ranking sandbox for the Redrob Intelligent Candidate Discovery challenge."
)

st.divider()

with st.expander("About this sandbox", expanded=False):
    st.markdown("""
This demo runs the same ranking pipeline (`rank.py`) used to produce the official submission, on
whatever candidate sample you provide. The pipeline applies, in order:

1. **Honeypot filter** -- removes profiles with internally inconsistent or implausible claims.
2. **Semantic recall** -- compares each profile against the target job description using
   sentence embeddings.
3. **Rubric scoring** -- rates fit against the job description's explicit requirements.
4. **Skill-trust correction** -- discounts listed skills that aren't corroborated by career history.
5. **Behavioral availability modifier** -- adjusts for engagement signals indicating openness to outreach.
6. **Reasoning generation** -- produces a short, factual explanation for each rank.

**Note on semantic recall:** the embeddings used in step 2 are precomputed offline over the full
candidate pool (no network access at ranking time, per the challenge's compute constraints). For
candidates outside that precomputed pool -- including any sample you upload here -- the semantic
term contributes zero, and ranking is driven by the rubric, skill-trust, and behavioral stages
instead. Candidates already in the precomputed pool, such as the bundled sample below, get the
full semantic comparison. Every other stage behaves identically to the production run.
        """)

st.subheader("1. Choose input data")

source_mode = st.radio(
    "Candidate data source",
    options=["Bundled sample dataset", "Upload my own file"],
    horizontal=True,
    help=(
        "The bundled sample is a small excerpt from the official challenge dataset "
        "(sample_candidates.json), already in the format the pipeline expects."
    ),
)

candidates: list[dict] = []
source = "-"


def _parse(raw: bytes) -> list[dict]:
    """Accept a JSON array, a single JSON object, or JSONL -- return a list of candidate dicts."""
    text = raw.decode("utf-8", "ignore").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return obj.get("candidates", [obj]) if "candidates" in obj else [obj]
    except json.JSONDecodeError:
        pass
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


if source_mode == "Bundled sample dataset":
    candidates = _parse(SAMPLE.read_bytes()) if SAMPLE.exists() else []
    source = f"Bundled sample dataset -- {len(candidates)} candidates from sample_candidates.json"
    st.info(
        "Using the bundled sample dataset. It ships with this repository at "
        "`sample_candidates.json` and is a representative excerpt of the official candidate "
        "pool, ready to rank without any upload."
    )
else:
    uploaded = st.file_uploader(
        "Candidate sample file (.json array, single object, or .jsonl), up to 100 candidates",
        type=["json", "jsonl"],
    )
    if uploaded is not None:
        candidates = _parse(uploaded.getvalue())
        source = f"{uploaded.name} -- {len(candidates)} candidates"

if candidates and len(candidates) > 100:
    st.warning(
        f"Sample has {len(candidates)} candidates; the sandbox caps at 100. Using the first 100."
    )
    candidates = candidates[:100]

st.subheader("2. Configure and run")

col1, col2 = st.columns([1, 1])
with col1:
    top_n = st.number_input(
        "Number of top candidates to return",
        min_value=1,
        max_value=100,
        value=20,
        step=5,
    )
with col2:
    st.metric("Candidates loaded", len(candidates) if candidates else 0)

run = st.button("Rank candidates", type="primary", disabled=not candidates)

st.divider()
st.subheader("3. Results")

if run:
    n_want = min(int(top_n), len(candidates))
    with st.spinner(f"Ranking {len(candidates)} candidates, returning top {n_want}..."):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for c in candidates:
                tmp.write(json.dumps(c) + "\n")
            tmp_path = tmp.name
        ranked = select_top(tmp_path, n_want)

    st.success(f"Ranked {len(ranked)} candidates from: {source}.")

    rows = [
        {
            "rank": i,
            "candidate_id": r["candidate_id"],
            "score": round(r["score"], 4),
            "title": r["f"]["current_title"],
            "years_experience": r["f"]["years_of_experience"],
            "location": r["f"]["location"],
            "reasoning": r["reasoning"],
        }
        for i, r in enumerate(ranked, start=1)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    buf = io.StringIO()
    tmp_csv = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    write_submission(ranked, tmp_csv.name)
    buf.write(Path(tmp_csv.name).read_text(encoding="utf-8"))
    st.download_button(
        "Download ranked CSV (submission format)",
        buf.getvalue(),
        file_name="submission_sample.csv",
        mime="text/csv",
    )
else:
    st.write(
        "Select a data source above and click **Rank candidates** to see results here."
    )
