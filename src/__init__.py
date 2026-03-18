"""
src — merchant issue triage RAG system.

Package structure:
    src.config          — centralized configuration (paths, model names, tuning knobs)
    src.ingest          — document loading, chunking, embedding, vector store write
    src.retriever       — semantic search with metadata filtering and MMR
    src.triage_agent    — LangGraph triage agent and TriageResponse schema
    src.evaluate        — LangSmith evaluation suite and golden dataset
"""
