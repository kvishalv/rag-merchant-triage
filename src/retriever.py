"""
retriever.py
------------
Semantic search over the embedded merchant issue vector store.

Supports:
  - Pure similarity search (cosine distance on OpenAI embeddings)
  - Metadata filtering by severity, merchant_tier, and/or component
  - MMR (Maximal Marginal Relevance) retrieval to surface diverse results
    when the query could match multiple related but distinct issue patterns

This module is consumed directly by triage_agent.py but can also be used
standalone for exploratory retrieval during debugging or eval set construction.

Usage:
    from src.retriever import build_retriever, search

    retriever = build_retriever()
    results = search(retriever, "payment failures on Visa cards", severity="P0")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ingest import CHROMA_PERSIST_DIR, COLLECTION_NAME

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default number of documents to retrieve per query. In production this
# was tuned to k=6 — enough context for the LLM to reason across multiple
# similar past cases without blowing the context window.
DEFAULT_K = 6

# Fetch fetch_k candidates then re-rank with MMR when diversity mode is on.
# Higher fetch_k increases result diversity at the cost of latency.
MMR_FETCH_K = 20
MMR_LAMBDA = 0.6  # 0 = max diversity, 1 = max relevance; 0.6 balances both


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single retrieved issue with its similarity score and parsed metadata.

    score is the cosine similarity (0–1, higher = more similar). Note that
    ChromaDB returns distance (lower = more similar) internally; we invert
    it here so callers always work with a consistent "higher is better" score.
    """
    issue_id: str
    title: str
    component: str
    severity: str
    merchant_tier: str
    tags: list[str]
    content: str
    score: float
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_document(cls, doc: Document, score: float) -> "RetrievalResult":
        meta = doc.metadata
        tags_raw = meta.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
        return cls(
            issue_id=meta.get("issue_id", "unknown"),
            title=meta.get("title", ""),
            component=meta.get("component", ""),
            severity=meta.get("severity", ""),
            merchant_tier=meta.get("merchant_tier", ""),
            tags=tags,
            content=doc.page_content,
            score=score,
            chunk_index=meta.get("chunk_index", 0),
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Vector store loading
# ---------------------------------------------------------------------------

def load_vector_store(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Load an existing ChromaDB collection from disk.

    Raises a clear error if the collection doesn't exist yet so operators
    know to run ingest.py first rather than silently returning empty results.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB persistence directory not found at {persist_dir}. "
            "Run `python src/ingest.py` to populate the vector store first."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Verify the collection isn't empty — a common footgun after a reset
    count = vector_store._collection.count()
    if count == 0:
        raise RuntimeError(
            f"Collection '{collection_name}' is empty. "
            "Run `python src/ingest.py` to ingest documents."
        )

    logger.info("Loaded collection '%s' with %d vectors.", collection_name, count)
    return vector_store


# ---------------------------------------------------------------------------
# Metadata filter construction
# ---------------------------------------------------------------------------

def build_where_filter(
    severity: Optional[str] = None,
    merchant_tier: Optional[str] = None,
    component: Optional[str] = None,
) -> Optional[dict]:
    """
    Build a ChromaDB $and/$eq metadata filter from optional kwargs.

    ChromaDB's 'where' clause uses MongoDB-style operators. When multiple
    filters are provided they are combined with $and. Returns None if no
    filters are specified (unfiltered search).

    Examples:
        build_where_filter(severity="P0")
        -> {"severity": {"$eq": "P0"}}

        build_where_filter(severity="P1", component="webhooks")
        -> {"$and": [{"severity": {"$eq": "P1"}}, {"component": {"$eq": "webhooks"}}]}
    """
    conditions = []
    if severity:
        conditions.append({"severity": {"$eq": severity}})
    if merchant_tier:
        conditions.append({"merchant_tier": {"$eq": merchant_tier}})
    if component:
        conditions.append({"component": {"$eq": component}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def search(
    vector_store: Chroma,
    query: str,
    k: int = DEFAULT_K,
    severity: Optional[str] = None,
    merchant_tier: Optional[str] = None,
    component: Optional[str] = None,
    use_mmr: bool = False,
) -> list[RetrievalResult]:
    """
    Run a semantic search query against the merchant issue vector store.

    Args:
        vector_store: A loaded Chroma instance (from load_vector_store).
        query: Natural language description of the incoming issue.
        k: Number of results to return.
        severity: Optional metadata filter (e.g., "P0", "P1").
        merchant_tier: Optional metadata filter (e.g., "enterprise", "smb").
        component: Optional metadata filter (e.g., "payment_processing", "webhooks").
        use_mmr: If True, use Maximal Marginal Relevance to diversify results.
                 Useful when the issue description is broad and you want coverage
                 across different root cause patterns rather than the top-k most
                 similar documents.

    Returns:
        List of RetrievalResult objects sorted by descending relevance score.
    """
    where_filter = build_where_filter(severity, merchant_tier, component)

    if use_mmr:
        # MMR doesn't return scores natively; we use 1.0 as a placeholder
        docs = vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=MMR_FETCH_K,
            lambda_mult=MMR_LAMBDA,
            filter=where_filter,
        )
        results = [RetrievalResult.from_document(doc, score=1.0) for doc in docs]
    else:
        docs_and_scores = vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=where_filter,
        )
        results = [
            RetrievalResult.from_document(doc, score=score)
            for doc, score in docs_and_scores
        ]
        # Sort highest relevance first
        results.sort(key=lambda r: r.score, reverse=True)

    logger.info(
        "Query '%s...' returned %d results (filters: severity=%s, tier=%s, component=%s)",
        query[:60],
        len(results),
        severity,
        merchant_tier,
        component,
    )
    return results


def deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """
    Remove duplicate issue_ids from retrieval results, keeping the highest-
    scoring chunk per issue. This is necessary because a single issue can
    produce multiple chunks, all of which may rank highly for a given query.

    The triage agent works best with distinct cases, not repeated chunks
    from the same issue.
    """
    seen: dict[str, RetrievalResult] = {}
    for result in results:
        if result.issue_id not in seen or result.score > seen[result.issue_id].score:
            seen[result.issue_id] = result
    return list(seen.values())


def format_results_for_context(results: list[RetrievalResult]) -> str:
    """
    Serialize retrieved results into a compact, structured string for injection
    into the LLM prompt as few-shot context.

    Format is intentionally verbose enough for the LLM to reason about root
    causes and resolutions, while staying within ~3k tokens for k=6 results.
    """
    if not results:
        return "No relevant past cases found."

    sections = []
    for i, r in enumerate(results, start=1):
        section = (
            f"--- Case {i}: {r.issue_id} (severity={r.severity}, "
            f"component={r.component}, tier={r.merchant_tier}) ---\n"
            f"{r.content}"
        )
        sections.append(section)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Retriever factory for LangChain integration
# ---------------------------------------------------------------------------

def build_retriever(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    k: int = DEFAULT_K,
    severity: Optional[str] = None,
    merchant_tier: Optional[str] = None,
    component: Optional[str] = None,
):
    """
    Build a LangChain-compatible retriever from the loaded vector store.

    Returns a Chroma retriever configured with the given metadata filters and k.
    This object is compatible with LangChain LCEL (|) composition, so it can
    be dropped directly into a RunnableSequence as the retrieval step.

    The returned retriever uses similarity search (not MMR) by default, which
    is preferred for the triage agent since precision matters more than diversity
    when we're trying to match an incoming issue to its most likely root cause.
    """
    vector_store = load_vector_store(persist_dir, collection_name)
    where_filter = build_where_filter(severity, merchant_tier, component)

    search_kwargs: dict = {"k": k}
    if where_filter:
        search_kwargs["filter"] = where_filter

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
