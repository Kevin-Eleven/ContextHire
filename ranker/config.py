"""JD-derived constants and tunable weights.
Everything here encodes the *fixed* job description (Senior AI Engineer — Founding Team, Redrob AI). All domain knowledge is kept here.
"""

from __future__ import annotations

import datetime as _dt

REFERENCE_DATE = _dt.date(2026, 6, 21)

# Default input/output the reproduce_command points at.
DEFAULT_CANDIDATES = "candidates.jsonl"
DEFAULT_OUTPUT = "submission.csv"
TOP_N = 100

# --- Experience band (JD: "5-9 years", ideal 6-8)
EXP_BAND = (5.0, 9.0)
EXP_IDEAL = (6.0, 8.0)


TARGET_CITIES = {
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "new delhi",
    "gurgaon",
    "gurugram",
    "bengaluru",
    "bangalore",
    "delhi ncr",
    "ncr",
}

# JD down-weights pure-services careers
CONSULTING_FIRMS = {
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
    "mphasis",
    "deloitte",
    "ibm",
    "dxc",
    "persistent",
}


# Core "must have" key words from the JD's skills section.
KW_RETRIEVAL = {
    "embedding",
    "embeddings",
    "sentence-transformer",
    "sentence transformers",
    "bge",
    "e5",
    "retrieval",
    "semantic search",
    "vector search",
    "rag",
}
KW_VECTOR_DB = {
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "faiss",
    "vector database",
    "vector db",
    "hybrid search",
}
KW_RANKING_EVAL = {
    "ranking",
    "ranker",
    "learning to rank",
    "ltr",
    "recommendation",
    "recommender",
    "recsys",
    "ndcg",
    "mrr",
    "map",
    "a/b test",
    "ab test",
    "evaluation framework",
}
KW_ML_GENERAL = {
    "machine learning",
    "ml ",
    "nlp",
    "information retrieval",
    "ir ",
    "fine-tune",
    "fine-tuning",
    "lora",
    "qlora",
    "peft",
    "xgboost",
}
KW_NICE_TO_HAVE = {
    "lora",
    "qlora",
    "peft",
    "learning to rank",
    "xgboost",
    "hr-tech",
    "hrtech",
    "recruiting",
    "marketplace",
    "distributed",
    "inference optimization",
    "open source",
}

# Negative signal keywords
KW_RESEARCH_ONLY = {
    "research scientist",
    "phd researcher",
    "postdoc",
    "academic",
    "research lab",
}

KW_NON_FIT_DOMAIN = {
    "computer vision",
    "image classification",
    "speech recognition",
    "robotics",
    "autonomous",
    "lidar",
}
KW_FRAMEWORK_ENTHUSIAST = {"langchain", "llamaindex", "autogpt", "tutorial", "demo"}

# Titles that indicate the candidate stopped writing code jd down weights
NON_CODING_TITLES = {
    "architect",
    "tech lead",
    "engineering manager",
    "director",
    "vp",
    "head of",
}

# --- Component weights for the final blend
WEIGHTS = {
    "semantic": 0.35,  # embedding/BM25
    "rubric": 0.45,  #  JD rubric
    "skill_trust": 0.20,  #  Skills corroboration
}

BEHAVIORAL_MODIFIER_FLOOR = (
    0.5  # worst-case down-weight for the unavailable (never zero)
)
