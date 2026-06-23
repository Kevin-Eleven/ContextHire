# ContextHire — Redrob Senior AI Engineer Ranker

Ranks the top 100 of 100,000 candidate profiles against one fixed job description
(**Senior AI Engineer — Founding Team, Redrob AI**) and emits a validated submission CSV with a
human-readable reason for every pick.

The dataset is **adversarial by design**: the highest-density `skills[]` arrays belong to
keyword-stuffers, and ~80 "subtly impossible" honeypot profiles will disqualify a submission if more
than 10% of them reach the top 100. So this ranker treats `skills[]` as untrusted and lets a
candidate's actual **career history** carry the weight.

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Measured: **~31s wall-clock, ~55 MB peak RAM**, CPU-only, no network — well inside the 5-min / 16 GB
budget.

## Approach — an explainable hybrid cascade

```
100K candidates  (streamed line-by-line from candidates.jsonl)
  │
  ├─ Stage 0  Honeypot filter          drop internally-impossible profiles  (avoid the >10% DQ)
  │
  │   positive fit  =  w_sem · A  +  w_rub · B  +  w_trust · C        (config.WEIGHTS 0.35/0.45/0.20)
  │     ├─ Stage A  Semantic recall    bge-small dense cosine + BM25 vs the JD (precomputed offline)
  │     ├─ Stage B  JD rubric          role tier, retrieval/ranking evidence, 5-9y, pre-2022 ML, location
  │     └─ Stage C  Skill-trust        a claimed skill counts only if its name appears in career text
  │
  ├─ Stage B penalties  ×              consulting-only / non-coding title / wrong domain / job-hopping
  ├─ Stage D behavioral ×              notice period, open-to-work, activity, responsiveness (bounded, floor 0.5)
  │
  └─ Top 100  →  feature-driven reasoning  →  validate  →  submit
```

**Why these choices**

- **`skills[]` is adversarial.** The decisive fit signals come from `career_history` titles and
  descriptions plus `summary` — evidence of _building_ retrieval/ranking/recsys at product companies.
  A "Marketing Manager" with a perfect AI skill list lands in the bottom title tier (`scoring.py`).
- **Skill-trust, not skill-count.** A self-reported skill earns trust only to the degree the career
  text corroborates it (`skill_trust.py`) — a stuffer's "RAG / embeddings" claims never appear in
  their non-AI job descriptions, so they earn ~zero trust.
- **No vector DB, no learning-to-rank model.** At 100K with a single fixed JD, one dense matmul beats
  FAISS/Pinecone overhead, and with no labels a rule rubric is more honest and defensible than a
  trained ranker.
- **Behavioral signals are a bounded multiplier, never an addend** — availability breaks ties among
  similar-fit candidates but can never outrank genuine fit.
- **Every threshold is data-calibrated.** `calibrate.py` profiles the full pool so each cliff sits on
  a real distribution point (a median / p90 / band edge), not a guess — see `calibration_report.txt`.

## Repository layout

```
rank.py                  # entry point (the reproduce_command); candidates.jsonl → submission.csv
precompute_embeddings.py # OFFLINE, declared: writes data/*.npy (embeddings, ids, jd, bm25)
calibrate.py             # OFFLINE: measures the dataset constants every threshold is tuned to
ranker/
  io.py                  # streaming JSONL loader (orjson), O(1) memory in pool size
  features.py            # raw profile → normalized feature dict
  honeypot.py            # Stage 0 impossibility filter
  recall.py              # Stage A semantic index (numpy-only, loads data/*.npy)
  scoring.py             # Stage B JD rubric + bounded penalty factors
  skill_trust.py         # Stage C anti-stuffing corroboration
  behavioral.py          # Stage D availability multiplier
  reasoning.py           # feature-driven, rank-toned reasoning strings
  config.py              # JD-derived constants and blend weights
data/*.npy               # precomputed artifacts (~9.6 MB, shipped — ranking needs no network)
sanity.py                # local harness: prints top-30 + reasoning, gates on hard checks
job_description.md        # the JD, operationalized into the rubric
submission_metadata.yaml # portal metadata (declares pre_computation_required: true)
```

## Running it

```bash
# 1. Environment (ranking needs only numpy + orjson; the rest is offline-only)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. (offline, optional — artifacts are already committed) recompute embeddings + calibration
python precompute_embeddings.py --candidates ./candidates.jsonl
python calibrate.py            --candidates ./candidates.jsonl

# 3. Rank
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 4. Sanity-check, then validate
python sanity.py               --candidates ./candidates.jsonl
python validate_submission.py  submission.csv     # → "Submission is valid."
```

## Compute & reproducibility

| Constraint           | Budget  | Measured                      |
| -------------------- | ------- | ----------------------------- |
| Wall-clock (ranking) | ≤ 5 min | ~31 s on 100K                 |
| RAM                  | ≤ 16 GB | ~55 MB peak RSS (streaming)   |
| GPU                  | none    | CPU-only                      |
| Network during rank  | none    | none — `data/*.npy` are local |

Offline precomputation is **declared** in `submission_metadata.yaml`
(`pre_computation_required: true`, ~8 min): `precompute_embeddings.py` embeds the top-12000
candidates by rubric with `BAAI/bge-small-en-v1.5` and computes BM25 over their career text against
the fixed JD. `rank.py` loads only the resulting `data/*.npy` and runs fully offline — if those
artifacts are absent it degrades gracefully to a rubric-only ranking.
