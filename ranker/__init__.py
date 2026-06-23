"""Redrob candidate-ranking package.
Pipeline:
    io        -> stream candidates.jsonl(.gz) line-by-line
    features  -> structured features per candidate
    honeypot  -> Impossibility filter
    recall    -> Dense + BM25 hybrid recall
    scoring   -> Rubric + trust + behavioral
    reasoning -> feature-driven templated reasoning
"""
