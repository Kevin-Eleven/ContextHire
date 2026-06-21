"""Redrob candidate-ranking package.

Pipeline (see constraint.md and the plan):
    io        -> stream candidates.jsonl(.gz) line-by-line
    features  -> structured features per candidate
    honeypot  -> Stage 0 impossibility filter           (Stage 2)
    recall    -> Stage A dense + BM25 hybrid recall      (Stage 5)
    scoring   -> Stage B/C/D rubric + trust + behavioral (Stages 3,4,6)
    reasoning -> feature-driven templated reasoning      (Stage 6)
"""
