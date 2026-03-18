"""
config.py
---------
Central configuration for the merchant issue triage system.

All path constants, model identifiers, and tuning parameters live here.
Import from this module rather than defining constants inline in other modules.
This makes it trivial to swap models, adjust chunk sizes, or point at a
different vector store without hunting through multiple files.

Environment variable overrides are supported for the values most likely
to differ between local dev, CI, and production.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is two levels up from this file (src/config.py → src/ → root)
PROJECT_ROOT = Path(__file__).parent.parent

FIXTURES_PATH = PROJECT_ROOT / "data" / "fixtures" / "merchant_issues.json"

# ChromaDB local persistence directory.
# Override via CHROMA_PERSIST_DIR env var (e.g., for CI or Docker volumes).
CHROMA_PERSIST_DIR = Path(
    os.environ.get("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / ".chroma_db"))
)

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

COLLECTION_NAME = "merchant_issues"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

# text-embedding-3-small: 1536 dims, strong price/perf for retrieval.
# Swap to text-embedding-3-large for higher accuracy at ~6x cost.
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# LLM (triage agent)
# ---------------------------------------------------------------------------

# GPT-4o for triage reasoning and LLM-as-judge evaluation.
TRIAGE_MODEL = os.environ.get("OPENAI_TRIAGE_MODEL", "gpt-4o")

# Low temperature keeps triage responses grounded and reproducible.
# Do not raise above 0.3 for triage — it increases hallucination of root causes.
TRIAGE_TEMPERATURE = 0.1

# Judge model for evaluate.py. Must be capable of nuanced scoring.
JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o")
JUDGE_TEMPERATURE = 0.0  # Zero temp for deterministic scoring

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Tuned so each chunk fits a full issue segment (description + root_cause)
# without splitting mid-sentence. Overlap preserves cross-boundary context.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Separators tried in order — prefer semantic breaks over arbitrary character splits
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Default number of unique issues to pass to the triage LLM as context.
# Tuned to keep the combined context under ~3k tokens for k=5 with our fixture format.
DEFAULT_K = 6

# Candidate pool size for MMR re-ranking. Larger = more diverse results.
MMR_FETCH_K = 20

# MMR lambda: 0 = max diversity, 1 = max relevance.
# 0.6 balances coverage across different root cause patterns vs. precision.
MMR_LAMBDA = 0.6

# If filtered retrieval returns fewer than this many results, fall back to
# unfiltered search to prevent context starvation on narrow metadata filters.
MIN_FILTERED_RESULTS = 3

# Max unique issues passed to the triage LLM after deduplication.
MAX_CONTEXT_ISSUES = 5

# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------

LANGSMITH_DATASET_NAME = "merchant-triage-golden"
LANGSMITH_RETRIEVAL_EXPERIMENT_PREFIX = "retrieval"
LANGSMITH_TRIAGE_EXPERIMENT_PREFIX = "triage"

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# Fields required in every fixture record. Used by ingest.py validation.
REQUIRED_ISSUE_FIELDS = frozenset({
    "issue_id",
    "severity",
    "merchant_tier",
    "component",
    "title",
    "description",
    "root_cause",
    "resolution",
})

# Valid severity levels (used for metadata filter validation)
VALID_SEVERITIES = frozenset({"P0", "P1", "P2", "P3"})

# Valid merchant tiers
VALID_MERCHANT_TIERS = frozenset({"enterprise", "mid-market", "smb"})

# Valid component values — must match what the LLM is prompted to use
VALID_COMPONENTS = frozenset({
    "payment_processing",
    "webhooks",
    "subscription_billing",
    "payout",
    "fraud_risk",
    "other",
})
