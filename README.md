# rag-merchant-triage

An anonymized, simplified version of a production Gen AI knowledge platform built to accelerate the diagnosis of high-business-impact merchant issues on a payments infrastructure.

The original system operates across millions of transactions daily, where the difference between a 5-minute and 50-minute time-to-diagnosis on a P0 can mean hundreds of thousands of dollars in failed payment volume and SLA penalties for enterprise merchants. This repo isolates the core RAG pipeline — ingestion, retrieval, triage agent, and eval loop — as a standalone reference implementation.

---

## What This Solves

Payments platforms accumulate a dense history of resolved incidents: webhook delivery failures, processor routing regressions, subscription billing edge cases, fraud model misfires. The pattern recognition work that senior engineers do when a new incident arrives — "this looks like that FX cache issue from Q3" — is exactly the kind of work a well-built RAG system can handle at 4 AM when the senior engineer is asleep.

The specific problems this addresses:

**Mean time-to-diagnosis is dominated by search, not reasoning.** On-call engineers spend the majority of their investigation time looking for whether this has happened before, not figuring out what to do once they have context. Semantic search over past incidents collapses this.

**Incident knowledge is scattered and unstructured.** Postmortems live in Confluence. Slack threads have the real detail. Jira tickets have partial root causes. This system treats the combined incident record as a corpus and makes it queryable.

**Triage quality is inconsistent across engineers and shifts.** A structured output (root cause hypothesis, affected component, recommended action, confidence score) gives every on-call engineer — regardless of their familiarity with the specific subsystem — a grounded starting point rather than a blank page.

**Alert fatigue causes slow escalation decisions.** The escalation signal built into the triage response (informed by whether similar prior incidents were P0-severity with high transaction impact) gives a second opinion on severity before a human makes the call.

---

## Architecture

```
Issue Intake (free-text description)
        │
        ▼
  Metadata Hints
  (severity, tier,
   component)
        │
        ▼
┌───────────────────┐
│  OpenAI Embedding │  ← text-embedding-3-small
│  (query vector)   │
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────────┐
│           ChromaDB Vector Store        │
│  (embedded issue chunks + metadata)    │
│                                        │
│  Metadata filters: severity,           │
│  merchant_tier, component              │
│                                        │
│  Fallback: unfiltered search if        │
│  filtered results < 3                  │
└────────────┬───────────────────────────┘
             │  top-k similar chunks
             ▼
┌────────────────────────────────────────┐
│   Deduplication + Context Assembly     │
│   (collapse multi-chunk issues,        │
│    format for LLM context window)      │
└────────────┬───────────────────────────┘
             │  structured context string
             ▼
┌────────────────────────────────────────┐
│          LangGraph Agent               │
│                                        │
│  Node 1: retrieve                      │
│  Node 2: triage (GPT-4o + LCEL chain) │
│                                        │
│  Output: TriageResponse (Pydantic)     │
│  • likely_root_cause                   │
│  • affected_component                  │
│  • recommended_action                  │
│  • confidence_score (0.0–1.0)          │
│  • escalation_recommended              │
│  • supporting_cases [ ]                │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│         LangSmith Eval Loop            │
│                                        │
│  Retrieval eval: recall@k              │
│  Triage eval: component accuracy,      │
│               root cause quality       │
│               (LLM-as-judge),          │
│               escalation accuracy      │
│                                        │
│  Golden dataset: 9 hand-curated        │
│  examples, version-controlled in code  │
└────────────────────────────────────────┘
```

See `ARCHITECTURE.md` for the extended diagram with component boundaries and data flow annotations.

---

## Stack

**Language:** Python 3.11+

**Orchestration:** LangGraph (StateGraph) for the agent graph, LangChain LCEL for the triage chain composition.

**LLM:** OpenAI GPT-4o (triage reasoning, LLM-as-judge evaluation). Temperature 0.1 for triage to keep responses grounded; temperature 0 for the judge to minimize scoring variance.

**Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions). Chosen over `text-embedding-3-large` for the retrieval use case here — the quality difference is marginal for domain-specific incident text, and the 6x cost difference matters at scale.

**Vector store:** ChromaDB with local persistence. The production equivalent uses pgvector on a managed Postgres instance (Aurora), which simplifies ops by keeping the vector store co-located with the relational incident data. ChromaDB is used here for zero-infrastructure setup.

**Evaluation:** LangSmith `evaluate()` with three custom evaluators — retrieval recall, component accuracy (rule-based), and root cause quality (LLM-as-judge). The golden dataset is maintained as code in `evaluate.py`.

**Schema validation:** Pydantic v2 for `TriageResponse` and `SupportingCase`. Using structured outputs at the LLM boundary means downstream consumers (incident trackers, PagerDuty enrichment) always get typed data, not raw strings.

---

## Project Structure

```
rag-merchant-triage/
├── src/
│   ├── ingest.py           # Document loading, chunking, embedding, ChromaDB write
│   ├── retriever.py        # Semantic search, metadata filtering, MMR, result formatting
│   ├── triage_agent.py     # LangGraph agent, TriageResponse schema, CLI
│   └── evaluate.py         # LangSmith eval suite, golden dataset, evaluators
├── data/
│   └── fixtures/
│       └── merchant_issues.json   # 15 anonymized merchant issue records
├── ARCHITECTURE.md         # Extended text-based architecture diagram
├── .env.example            # Environment variable template
├── requirements.txt        # Pinned dependencies
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.11+, an OpenAI API key, a LangSmith API key (free tier works for evals).

```bash
# Clone and install
git clone https://github.com/your-org/rag-merchant-triage.git
cd rag-merchant-triage
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and populate OPENAI_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

# Ingest the fixture data into ChromaDB
python src/ingest.py

# Run a triage query
python src/triage_agent.py

# Run evaluations (requires LangSmith credentials)
python src/evaluate.py --push-dataset --eval all
```

---

## Usage

**Ingest:**
```bash
python src/ingest.py --reset   # wipe and re-ingest from scratch
```

**Triage a new issue:**
```python
from src.triage_agent import run_triage

result = run_triage(
    issue_description="EUR transactions are all failing with 500. Started 10 minutes ago.",
    severity="P0",
    merchant_tier="enterprise",
)
print(result.likely_root_cause)
print(result.recommended_action)
print(f"Confidence: {result.confidence_score:.0%}")
print(f"Escalate: {result.escalation_recommended}")
```

**Retrieval only:**
```python
from src.retriever import build_retriever, load_vector_store, search

vs = load_vector_store()
results = search(vs, "webhook events not delivering", component="webhooks", k=5)
for r in results:
    print(r.issue_id, r.score, r.severity)
```

**Evaluate:**
```bash
# Retrieval recall only
python src/evaluate.py --eval retrieval

# Full triage quality eval (costs ~$0.05 in LLM calls for 9 examples)
python src/evaluate.py --eval triage
```

---

## Design Decisions

**Why LangGraph instead of a plain LCEL chain?**
The current graph is linear (retrieve → triage → END), which could be an LCEL pipe. LangGraph is used because the graph is the extension point — the next logical nodes are a human escalation confirmation step for P0s, and a parallel node that enriches context with live processor status page data. Restructuring a chain into a graph mid-production is painful; starting as a graph costs nothing.

**Why separate retriever.py from triage_agent.py?**
The retriever is independently testable and useful for retrieval eval, exploratory search during on-call, and dataset construction. Keeping it decoupled from the agent means it can be called directly without spinning up the full graph.

**Why is the golden dataset in code rather than a LangSmith dataset?**
Incident triage ground truth changes as the platform evolves. Having the dataset in a Python file means it's part of the PR review process — a new incident pattern added to the fixtures must also have a corresponding golden example added to the eval set. This enforces eval coverage at the contribution level.

**Why confidence_score rather than just returning the top result?**
The retrieval similarity score isn't exposed directly to callers because it doesn't account for reasoning quality — a high cosine similarity to a loosely related past case can produce a confident but wrong triage. The LLM-generated confidence score reflects the agent's own assessment of how well the retrieved cases actually explain the incoming issue.

---

## Limitations

This is a simplified reference implementation. Production differences include pgvector on Postgres (not ChromaDB), integration with PagerDuty and Jira for structured output consumption, a feedback loop where confirmed root causes are re-ingested to improve future retrieval, and a larger fixture corpus (~800 anonymized incidents vs. 15 here).

The evaluation dataset is small by design. In production, the golden dataset was grown iteratively over several months as new incident patterns were confirmed by senior engineers and added to the eval set.
