# rag-merchant-triage Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Prerequisites: Python 3.11+, a populated .env file (copy from .env.example).
#
# Quick start:
#   make install      → install deps in editable mode
#   make ingest       → embed fixtures and populate ChromaDB
#   make triage       → run a sample triage query
#   make eval         → run all LangSmith evaluations
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install ingest ingest-reset triage search eval eval-retrieval eval-triage \
        push-dataset lint clean clean-db

# Default target
help:
	@echo ""
	@echo "  rag-merchant-triage"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install all dependencies (editable mode)"
	@echo ""
	@echo "  Data pipeline"
	@echo "    make ingest           Embed fixtures and write to ChromaDB"
	@echo "    make ingest-reset     Wipe ChromaDB collection and re-ingest"
	@echo ""
	@echo "  Run"
	@echo "    make triage           Run a sample triage query"
	@echo "    make search Q=\"...\"   Semantic search (e.g. make search Q=\"webhook delay\")"
	@echo ""
	@echo "  Evaluation"
	@echo "    make push-dataset     Push golden dataset to LangSmith"
	@echo "    make eval             Run all evaluations (retrieval + triage)"
	@echo "    make eval-retrieval   Run retrieval recall eval only"
	@echo "    make eval-triage      Run triage quality eval only"
	@echo ""
	@echo "  Maintenance"
	@echo "    make lint             Run ruff linter"
	@echo "    make clean            Remove __pycache__ and .pyc files"
	@echo "    make clean-db         Delete ChromaDB persistence directory"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]" 2>/dev/null || pip install -e .
	@echo ""
	@echo "  Installed. Next: copy .env.example to .env and add your API keys."
	@echo "  Then run: make ingest"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

ingest:
	@echo "→ Ingesting fixtures into ChromaDB..."
	python -m src.ingest
	@echo "Done. Run 'make triage' to test retrieval."

ingest-reset:
	@echo "→ Wiping ChromaDB collection and re-ingesting..."
	python -m src.ingest --reset
	@echo "Done."

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

triage:
	@echo "→ Running sample triage query..."
	python -m src.triage_agent

# Usage: make search Q="webhook delivery delay"
search:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make search Q=\"your query here\""; \
		exit 1; \
	fi
	python -m src.retriever "$(Q)"

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (requires LANGCHAIN_API_KEY in .env)
# ─────────────────────────────────────────────────────────────────────────────

push-dataset:
	@echo "→ Pushing golden dataset to LangSmith..."
	python -m src.evaluate --push-dataset --eval retrieval  # push only, skip eval
	@echo "Dataset pushed."

eval:
	@echo "→ Running all evaluations (retrieval + triage)..."
	@echo "  Note: triage eval costs ~\$$0.05 in LLM calls for 9 examples."
	python -m src.evaluate --push-dataset --eval all

eval-retrieval:
	@echo "→ Running retrieval recall evaluation..."
	python -m src.evaluate --eval retrieval

eval-triage:
	@echo "→ Running triage quality evaluation..."
	python -m src.evaluate --eval triage

# ─────────────────────────────────────────────────────────────────────────────
# Maintenance
# ─────────────────────────────────────────────────────────────────────────────

lint:
	@command -v ruff >/dev/null 2>&1 || pip install ruff -q
	ruff check src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cleaned."

clean-db:
	@echo "→ Deleting ChromaDB persistence directory (.chroma_db/)..."
	rm -rf .chroma_db/
	@echo "Done. Run 'make ingest' to rebuild."
