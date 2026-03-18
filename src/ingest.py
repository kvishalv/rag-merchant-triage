"""
ingest.py
---------
Loads merchant issue documents from JSON fixtures, chunks them into
semantically coherent segments, embeds them via OpenAI, and persists
the resulting vectors (with metadata) into a ChromaDB collection.

Run this once to populate the vector store before using retriever.py
or triage_agent.py.

Usage:
    python -m src.ingest
    python -m src.ingest --fixtures data/fixtures/merchant_issues.json
    python -m src.ingest --reset   # wipe and re-ingest
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    FIXTURES_PATH,
    REQUIRED_ISSUE_FIELDS,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def build_document_text(issue: dict[str, Any]) -> str:
    """
    Construct the text representation of an issue that will be embedded.

    We deliberately concatenate the fields that carry the most diagnostic
    signal: title, description, root_cause, and resolution. Metadata
    (severity, merchant_tier, component, tags) is stored separately as
    filterable fields on the Chroma document — not embedded — so that
    retrieval can be both semantic AND metadata-filtered without polluting
    the embedding space with categorical tokens.
    """
    sections = [
        f"Issue: {issue['title']}",
        f"Description: {issue['description']}",
        f"Root Cause: {issue['root_cause']}",
        f"Resolution: {issue['resolution']}",
    ]
    return "\n\n".join(sections)


def build_metadata(issue: dict[str, Any]) -> dict[str, Any]:
    """
    Extract filterable metadata fields from an issue record.

    All values must be scalar (str, int, float, bool) for ChromaDB
    compatibility — lists are serialized to comma-separated strings.

    Note: 'title' is stored here so RetrievalResult can surface it
    without re-parsing the full document text.
    """
    return {
        "issue_id": issue["issue_id"],
        "title": issue["title"],           # stored for display; not for embedding
        "severity": issue["severity"],
        "merchant_tier": issue["merchant_tier"],
        "component": issue["component"],
        "tags": ",".join(issue.get("tags", [])),  # ChromaDB requires scalar metadata
        "ttd_minutes": issue.get("ttd_minutes", -1),
        "ttr_minutes": issue.get("ttr_minutes", -1),
        "affected_transactions": issue.get("affected_transactions", 0),
        "created_at": issue.get("created_at", ""),
    }


def load_issues(fixtures_path: Path = FIXTURES_PATH) -> list[dict[str, Any]]:
    """Load and validate the raw issue fixture JSON."""
    if not fixtures_path.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")

    with open(fixtures_path, "r", encoding="utf-8") as f:
        issues = json.load(f)

    for issue in issues:
        missing = REQUIRED_ISSUE_FIELDS - set(issue.keys())
        if missing:
            raise ValueError(
                f"Issue {issue.get('issue_id', '?')} missing required fields: {missing}"
            )

    logger.info("Loaded %d issues from %s", len(issues), fixtures_path)
    return issues


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def issues_to_documents(issues: list[dict[str, Any]]) -> list[Document]:
    """
    Convert issue dicts into LangChain Documents, then chunk them.

    Each issue becomes one or more Document chunks. Metadata is propagated
    to every chunk so retrieval results can always be filtered/sorted by
    severity, tier, and component regardless of which chunk matched.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    raw_docs: list[Document] = []
    for issue in issues:
        text = build_document_text(issue)
        metadata = build_metadata(issue)
        raw_docs.append(Document(page_content=text, metadata=metadata))

    chunks = splitter.split_documents(raw_docs)

    # Tag each chunk with its position within the parent issue for debugging
    chunk_counters: dict[str, int] = {}
    for chunk in chunks:
        issue_id = chunk.metadata["issue_id"]
        chunk_counters[issue_id] = chunk_counters.get(issue_id, 0) + 1
        chunk.metadata["chunk_index"] = chunk_counters[issue_id] - 1

    logger.info(
        "Produced %d chunks from %d issues (avg %.1f chunks/issue)",
        len(chunks),
        len(issues),
        len(chunks) / max(len(issues), 1),
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedding and storage
# ---------------------------------------------------------------------------

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Instantiate the OpenAI embeddings model.

    Uses EMBEDDING_MODEL from config (default: text-embedding-3-small).
    Swap to text-embedding-3-large in .env for higher accuracy at ~6x cost.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and populate it."
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )


def ingest(
    fixtures_path: Path = FIXTURES_PATH,
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> Chroma:
    """
    Full ingest pipeline: load → validate → chunk → embed → store.

    Args:
        fixtures_path: Path to the merchant issues JSON fixture file.
        persist_dir: Directory where ChromaDB will persist its data.
        collection_name: Name of the Chroma collection to write to.
        reset: If True, delete and recreate the collection before ingesting.

    Returns:
        The populated Chroma vector store instance.
    """
    issues = load_issues(fixtures_path)
    chunks = issues_to_documents(issues)
    embeddings = get_embedding_model()

    persist_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        import chromadb
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception:
            pass  # Collection didn't exist — no-op

    logger.info(
        "Embedding %d chunks into collection '%s' at %s ...",
        len(chunks),
        collection_name,
        persist_dir,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

    logger.info("Ingest complete. %d vectors written.", len(chunks))
    return vector_store


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest merchant issue documents into ChromaDB."
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=FIXTURES_PATH,
        help="Path to the merchant issues JSON fixture file.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=CHROMA_PERSIST_DIR,
        help="Directory for ChromaDB persistence.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before ingesting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(
        fixtures_path=args.fixtures,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        reset=args.reset,
    )
