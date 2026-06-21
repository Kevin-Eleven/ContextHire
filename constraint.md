# Constraints — Redrob "Intelligent Candidate Discovery & Ranking Challenge"

Every constraint the solution must satisfy, gathered from `problemStatement.txt`,
`job_description.docx`, `submission_spec.docx`, `README.docx`, `redrob_signals_doc.docx`,
`candidate_schema.json`, `submission_metadata_template.yaml`, and `validate_submission.py`.
Treat this as the checklist the build must never violate.

---

## 1. The task (fixed scope)

- Rank candidates from `India_runs_data_and_ai_challenge/candidates.jsonl` against **one fixed job
  description**: "Senior AI Engineer — Founding Team" at Redrob AI (`job_description.docx`).
- Output the **top 100 only**, best-fit first (rank 1 = best). Do **not** rank candidate 101 onward.
- The dataset is **100,000 candidates**, one JSON profile per line, ~465MB uncompressed
  (~52MB gzipped). **Stream it line-by-line — never load the whole file into memory.**

---

## 2. Compute constraints (hard — enforced in a sandboxed Docker at Stage 3)

The **ranking step** that produces the submission CSV must satisfy:

| Constraint    | Limit                                                             |
| ------------- | ----------------------------------------------------------------- |
| Total runtime | **≤ 5 minutes** wall-clock                                        |
| Memory        | **≤ 16 GB RAM**                                                   |
| Compute       | **CPU only — no GPU during ranking**                              |
| Network       | **OFF** — no external/hosted API calls of any kind during ranking |
| Disk          | **≤ 5 GB** intermediate state                                     |

- **No hosted LLM APIs during ranking** — explicitly no OpenAI, Anthropic, Cohere, Gemini, or any
  hosted LLM service. Running an LLM call per candidate will not fit the budget.
- Reproduction: at Stage 3, top-N submissions are **re-run inside a Docker container matching these
  limits exactly**. If it can't reproduce within the limits → **disqualified**, regardless of score.
- Must run end-to-end on a 16GB CPU-only machine within 5 minutes **before** submitting.

### Offline precomputation (allowed, but must be declared)

- Precomputing embeddings / features / indexes **offline** (before ranking, with network) is allowed.
- It must be **declared** in `submission_metadata.yaml` (`pre_computation_required: true` and
  `pre_computation_time_minutes`).
- Precomputed artifacts (e.g. `embeddings.npy`) are loaded at ranking time — generating them is **not**
  part of the 5-minute budget. The no-network/no-GPU rules apply only to the ranking step.

---

## 3. Submission file format (auto-rejected by `validate_submission.py` if violated)

- A single **CSV** file, **UTF-8** encoded.
- Filename = your registered **participant ID** + `.csv` (e.g. `team_xxx.csv`).
- Header row exactly, in this order: `candidate_id,rank,score,reasoning`
- Exactly **100 data rows** (rows 2–101), plus the 1 header row.
- Columns:
  - `candidate_id` — string, must match `^CAND_[0-9]{7}$`, must exist in `candidates.jsonl`,
    each appears **exactly once** (no duplicates).
  - `rank` — integer **1–100**, each value used **exactly once**.
  - `score` — float, **monotonically non-increasing as rank increases** (score@1 ≥ score@2 ≥ … ≥ score@100). Ties allowed.
  - `reasoning` — string, optional but **strongly recommended** (graded at Stage 4).
- **Tie-breaking:** equal scores still need unique ranks; break ties by a secondary model signal **or by
  `candidate_id` ascending** (the validator enforces: for equal scores, lower `candidate_id` gets the better rank).
- **Always run `python India_runs_data_and_ai_challenge/validate_submission.py <id>.csv` before submitting.**

---

## 4. Disqualifiers and traps (data-level)

- **Honeypots:** the dataset contains ~80 honeypots — "subtly impossible profiles."
  **If >10% of your top 100 are honeypots → automatic disqualification.** (Target: 0 in top 100.)
- **Built-in keyword trap:** ranking by "most AI keywords in the `skills[]` section" is explicitly the
  wrong answer. The `skills[]` array is adversarial — do **not** trust it at face value.
- Other planted traps to handle: **keyword stuffers**, **plain-language Tier-5s** (great candidates who
  never use buzzwords), and **behavioral twins**.

---

## 5. Reasoning-column rules (graded at Stage 4 manual review)

10 random rows are sampled and each `reasoning` entry is checked for:

- **Specific facts** from the profile (years, current title, named skills, signal values).
- **JD connection** — ties to specific JD requirements, not generic praise.
- **Honest concerns** — acknowledges real gaps/risks where they exist.
- **No hallucination** — every claim must correspond to something actually in the profile. Inventing
  skills/employers/experience is a red flag.
- **Variation** — the sampled reasonings must be substantively different, not templated/name-inserted.
- **Rank consistency** — tone must match the rank (no glowing rank-95 or critical rank-5 reasoning).

Penalized: empty, all-identical, name-only templated, hallucinated, or rank-contradicting reasoning.

---

## 6. Scoring (how the ranking is judged — hidden ground truth)

`Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`

- **The top 10 carries half the weight.** P@10 measures fraction of top-10 that are "relevant" (tier 3+).
- Scored **once, after submissions close**. **No live leaderboard, no feedback during the competition.**

---

## 7. Submission process / logistics

- **At most 3 submissions total**; the **last valid** submission counts. Earlier ones are not preserved.
- Required submission artifacts:
  1. The ranked **CSV** (top 100).
  2. A **GitHub repo** — clean, complete, working code. Private OK if organizer access can be granted at Stage 3.
  3. A **deck/PPT converted to PDF** explaining the approach.
  4. **`submission_metadata.yaml`** (from the template) — team identity, GitHub repo, sandbox link,
     `reproduce_command`, compute environment, AI-tools declaration, methodology, declarations.
  5. A working **sandbox link** (HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, or Binder)
     where the ranker can be run on a small sample.
- `reproduce_command` must be the single command producing `submission.csv` from `candidates.jsonl`
  (e.g. `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`) and must run within the
  compute constraints in §2.
- **AI-tool use is allowed** but must be declared honestly in the metadata; declarations may be verified at
  Stage 5 against the code/interview. Declared use is **not** penalized.
- **Stage 5 interview:** top entrants must walk through and defend their architecture and design choices.

---

## 8. Domain rules from the JD (what "fit" means — the ranking rubric)

These are not format rules but define correctness; the ranker must encode them.

**Down-weight / reject (negative signals):**

- `current_title` / career that contradicts the role (e.g. "Marketing Manager" with a stuffed AI skill list).
- Pure research/academia with no production deployment.
- "AI experience" = only recent (<12 months) LangChain-on-OpenAI, with no pre-LLM-era ML production.
- Senior who hasn't written production code in ~18 months (moved to "architect"/"tech lead").
- Title-chasers / job-hoppers (avg tenure < ~1.5 years with title escalation).
- Framework enthusiasts (GitHub full of hot-framework demos/tutorials).
- Entirely consulting-firm careers (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, …) with no product experience.
- Computer-vision / speech / robotics primary, without NLP/IR exposure.
- Entirely closed-source/proprietary for 5+ years with no external validation (papers, talks, OSS).
- **Behavioral unavailability** — down-weight candidates who aren't really hireable: low
  `recruiter_response_rate`, stale `last_active_date`, not `open_to_work`, low completion rates.
  (A modifier, **not** the primary ranking axis.)

**Up-weight (positive signals):**

- 5–9 years experience (ideal 6–8), with 4–5 in applied ML/AI at **product** companies.
- Production embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings, …).
- Vector DB / hybrid search experience (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
- Strong Python; ranking-evaluation frameworks (NDCG, MRR, MAP, A/B testing).
- Shipped an end-to-end ranking/search/recommendation system to real users at scale.
- Pre-LLM-era ML production experience.
- Nice-to-haves: LoRA/QLoRA/PEFT fine-tuning, learning-to-rank, HR-tech/marketplace, distributed/inference, OSS.
- Location fit: Pune / Noida / Hyderabad / Mumbai / Delhi NCR, or `willing_to_relocate`. (No visa sponsorship; outside India case-by-case.)
- Short notice period (<30 days ideal; 30+ raises the bar).

---

## 9. Data-handling constraints

- Profiles follow `candidate_schema.json` (draft-07). Required top-level keys: `candidate_id`, `profile`,
  `career_history`, `education`, `skills`, `redrob_signals`. `certifications` and `languages` are optional.
- Some fields use sentinel values: `github_activity_score = -1` (no GitHub), `offer_acceptance_rate = -1`
  (no offer history), `end_date = null` (current role). Handle these explicitly — don't treat sentinels as real values.
- The validator handles both `.jsonl` and gzipped `.jsonl.gz`; the loader should too.
  </content>
