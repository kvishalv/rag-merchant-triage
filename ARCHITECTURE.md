# Architecture

## System Overview

This document describes the end-to-end data flow and component boundaries of the merchant issue triage RAG system. There are two distinct runtime paths: the **ingest path** (offline, run once or on schedule) and the **query path** (real-time, invoked per incoming issue).

---

## Ingest Path

Builds and maintains the vector knowledge base from historical incident records.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             INGEST PATH                                  │
│                        (offline / on-schedule)                           │
└──────────────────────────────────────────────────────────────────────────┘

  data/fixtures/
  merchant_issues.json
         │
         │  15 anonymized incident records
         │  (issue_id, severity, component, title,
         │   description, root_cause, resolution,
         │   tags, ttd_minutes, ttr_minutes)
         │
         ▼
  ┌─────────────────────────────┐
  │   ingest.py: load_issues()  │
  │                             │
  │   Validates required fields │
  │   per record. Fails loudly  │
  │   on schema errors.         │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─────────────────────────────┐
  │  ingest.py: build_document  │
  │  _text() + build_metadata() │
  │                             │
  │  Embeddable text:           │
  │    title + description +    │
  │    root_cause + resolution  │
  │                             │
  │  Metadata (scalar only,     │
  │  for ChromaDB filters):     │
  │    severity, merchant_tier, │
  │    component, tags (csv),   │
  │    ttd_minutes, ttr_minutes │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─────────────────────────────┐
  │  RecursiveCharacterText     │
  │  Splitter                   │
  │                             │
  │  chunk_size=800             │
  │  chunk_overlap=120          │
  │                             │
  │  Separators: paragraph,     │
  │  newline, sentence, word    │
  │                             │
  │  Metadata propagated to     │
  │  every chunk so retrieval   │
  │  filters work at chunk level│
  └──────────┬──────────────────┘
             │  N chunks per issue
             ▼
  ┌─────────────────────────────┐
  │  OpenAI Embeddings          │
  │  text-embedding-3-small     │
  │  (1536 dimensions)          │
  │                             │
  │  Batch API call for all     │
  │  chunks                     │
  └──────────┬──────────────────┘
             │  float[1536] per chunk
             ▼
  ┌─────────────────────────────┐
  │  ChromaDB                   │
  │  PersistentClient           │
  │  .chroma_db/ (local disk)   │
  │                             │
  │  Collection: merchant_issues│
  │                             │
  │  Stores: vector + text +    │
  │  metadata per chunk         │
  │                             │
  │  Production equivalent:     │
  │  pgvector on Aurora         │
  └─────────────────────────────┘
```

---

## Query / Triage Path

Real-time path invoked when a new merchant issue is reported. Runs as a LangGraph StateGraph.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             QUERY PATH                                   │
│                          (real-time, per issue)                          │
└──────────────────────────────────────────────────────────────────────────┘

  Caller (on-call engineer / incident automation)
         │
         │  run_triage(
         │    issue_description="...",  ← free text
         │    severity="P1",            ← optional hint
         │    merchant_tier="enterprise",
         │    component_hint="webhooks"
         │  )
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  LangGraph StateGraph: TriageState                      │
  │                                                         │
  │  State fields:                                          │
  │    issue_description, severity, merchant_tier,          │
  │    component_hint, retrieved_cases,                     │
  │    retrieval_context, triage_response, messages         │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         │  entry_point
                         ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Node 1: retrieve                                                    │
  │  (retriever.py)                                                      │
  │                                                                      │
  │  1. Embed query via text-embedding-3-small                           │
  │  2. Build ChromaDB metadata filter from state hints                  │
  │     (severity AND merchant_tier AND component — any combination)     │
  │  3. similarity_search_with_relevance_scores(query, k=8, filter=...)  │
  │  4. If filtered results < 3 → fall back to unfiltered search         │
  │     (prevents retrieval starvation on narrow filters)                │
  │  5. Merge and deduplicate by issue_id                                │
  │     (keep highest-scoring chunk per unique issue)                    │
  │  6. Truncate to top-5 unique cases                                   │
  │  7. format_results_for_context() → structured string for LLM prompt  │
  │                                                                      │
  │  Writes: retrieved_cases[], retrieval_context                        │
  └──────────────────────┬───────────────────────────────────────────────┘
                         │
                         │  edge: retrieve → triage
                         ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Node 2: triage                                                      │
  │  (triage_agent.py)                                                   │
  │                                                                      │
  │  LCEL Chain:                                                         │
  │    ChatPromptTemplate                                                │
  │      (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE)                          │
  │    │                                                                 │
  │    ▼                                                                 │
  │    ChatOpenAI(model="gpt-4o", temperature=0.1)                       │
  │    │                                                                 │
  │    ▼                                                                 │
  │    JsonOutputParser(pydantic_object=TriageResponse)                  │
  │    │                                                                 │
  │    ▼                                                                 │
  │    TriageResponse (Pydantic)                                         │
  │      • likely_root_cause: str                                        │
  │      • affected_component: str                                       │
  │      • recommended_action: str                                       │
  │      • confidence_score: float (0.0–1.0)                             │
  │      • escalation_recommended: bool                                  │
  │      • supporting_cases: SupportingCase[]                            │
  │      • caveats: str | None                                           │
  │                                                                      │
  │  Writes: triage_response                                             │
  └──────────────────────┬───────────────────────────────────────────────┘
                         │
                         │  edge: triage → END
                         ▼
  TriageResponse returned to caller
         │
         │  consumed by:
         ├── On-call engineer review
         ├── Jira ticket auto-population (affected_component, root_cause)
         └── PagerDuty alert enrichment (escalation_recommended)
```

---

## Evaluation Loop

Offline evaluation run against a golden dataset stored in `evaluate.py`.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          LANGSMITH EVAL LOOP                             │
│                             (offline / CI)                               │
└──────────────────────────────────────────────────────────────────────────┘

  evaluate.py: GOLDEN_DATASET (9 hand-curated examples)
         │
         │  push_dataset_to_langsmith()
         ▼
  LangSmith Dataset: "merchant-triage-golden"
         │
         ├──────────────────────────────────────────────┐
         │                                              │
         │  Retrieval Eval                              │  Triage Eval
         ▼                                              ▼
  retrieval_target()                            triage_target()
  search(query, k=6, filters)                   run_triage(description, ...)
         │                                              │
         ▼                                              ▼
  retrieval_relevance_evaluator                 component_accuracy_evaluator
  ┌─────────────────────────┐                   ┌───────────────────────────┐
  │  Recall@k               │                   │  Rule-based               │
  │  1.0: all expected IDs  │                   │  1.0: correct component   │
  │       retrieved         │                   │  0.0: wrong component     │
  │  0.5: some expected IDs │                   └───────────────────────────┘
  │  0.0: none retrieved    │
  └─────────────────────────┘                   root_cause_quality_evaluator
                                                 ┌───────────────────────────┐
                                                 │  LLM-as-judge (GPT-4o)    │
                                                 │  0.0–1.0 float score      │
                                                 │  Checks: specificity,     │
                                                 │  keyword coverage,        │
                                                 │  grounding                │
                                                 └───────────────────────────┘

                                                 escalation_accuracy_evaluator
                                                 ┌───────────────────────────┐
                                                 │  Binary                   │
                                                 │  1.0: escalation matches  │
                                                 │  0.0: mismatch            │
                                                 └───────────────────────────┘
         │                                              │
         └──────────────────┬───────────────────────────┘
                            ▼
                  LangSmith Experiment Results
                  (scores, traces, side-by-side comparison)
```

---

## Component Boundary Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE          │  RESPONSIBILITY                   │  EXTERNAL CALLS   │
├─────────────────────────────────────────────────────────────────────────┤
│  ingest.py       │  Load, chunk, embed, write DB     │  OpenAI Embed API │
│  retriever.py    │  Query, filter, format results    │  OpenAI Embed API │
│  triage_agent.py │  Orchestrate graph, call LLM      │  OpenAI Chat API  │
│  evaluate.py     │  Run evals, score outputs         │  OpenAI Chat API  │
│                  │                                   │  LangSmith API    │
├─────────────────────────────────────────────────────────────────────────┤
│  ChromaDB        │  Persist + query vectors          │  (local disk)     │
│  LangSmith       │  Trace all LangChain calls        │  (automatic)      │
│  Pydantic        │  Validate structured outputs      │  (in-process)     │
└─────────────────────────────────────────────────────────────────────────┘
```

LangSmith tracing is enabled automatically for all LangChain and LangGraph calls when `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT` are set in the environment. No additional instrumentation is required.
