"""
triage_agent.py
---------------
LangGraph-based merchant issue triage agent.

Given an incoming issue description (and optional metadata hints like severity
and merchant tier), the agent:
  1. Retrieves semantically similar past cases from the vector store
  2. Reasons over those cases to identify the likely root cause
  3. Returns a structured TriageResponse with: likely_root_cause,
     affected_component, recommended_action, confidence_score, and
     references to the supporting past cases

Architecture:
  - LangGraph StateGraph with two nodes: `retrieve` and `triage`
  - State is typed via TypedDict so every node has a clear contract
  - The triage node uses an LLM chain (LCEL) with a structured output parser
  - LangSmith tracing is enabled automatically via LANGCHAIN_API_KEY env var

Usage:
    from src.triage_agent import run_triage

    result = run_triage(
        issue_description="Customers on iOS 17 are seeing a blank screen at checkout.",
        severity="P1",
        merchant_tier="enterprise",
    )
    print(result.likely_root_cause)
    print(result.recommended_action)
"""

import logging
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from retriever import (
    RetrievalResult,
    build_where_filter,
    deduplicate_results,
    format_results_for_context,
    load_vector_store,
    search,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class SupportingCase(BaseModel):
    """A past case cited as evidence for the triage decision."""
    issue_id: str = Field(description="Issue ID from the knowledge base.")
    relevance_note: str = Field(description="One sentence explaining why this case is relevant.")


class TriageResponse(BaseModel):
    """
    Structured output of the triage agent.

    Designed to map directly to fields in an internal issue tracker (e.g.,
    Jira custom fields) so the response can be auto-populated on ticket creation.
    """
    likely_root_cause: str = Field(
        description="Concise hypothesis for the most probable root cause, "
                    "grounded in the retrieved similar cases."
    )
    affected_component: str = Field(
        description="The platform component most likely at fault. "
                    "One of: payment_processing, webhooks, subscription_billing, "
                    "payout, fraud_risk, other."
    )
    recommended_action: str = Field(
        description="Specific, actionable next step for the on-call engineer. "
                    "Should name a concrete artifact (config, job, service) to inspect."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the triage (0.0–1.0). Reflects how closely "
                    "the retrieved cases match the incoming issue. <0.5 means "
                    "the issue is novel and manual investigation is needed.",
    )
    escalation_recommended: bool = Field(
        description="True if the issue pattern matches prior P0/P1 incidents "
                    "with high affected transaction counts."
    )
    supporting_cases: list[SupportingCase] = Field(
        description="Past cases from the knowledge base that informed this triage.",
        max_items=4,
    )
    caveats: Optional[str] = Field(
        default=None,
        description="Any important caveats, e.g., if the issue description is "
                    "too vague or if the retrieved cases are only loosely related."
    )


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class TriageState(TypedDict):
    """
    State passed between nodes in the LangGraph graph.

    Using TypedDict keeps the state contract explicit — each node declares
    exactly what fields it reads and writes, which makes the graph easier
    to test and trace in LangSmith.
    """
    # Inputs (set by the caller before graph invocation)
    issue_description: str
    severity: Optional[str]
    merchant_tier: Optional[str]
    component_hint: Optional[str]

    # Populated by the `retrieve` node
    retrieved_cases: list[RetrievalResult]
    retrieval_context: str  # formatted string passed to the LLM

    # Populated by the `triage` node
    triage_response: Optional[TriageResponse]

    # Message history (used for LangSmith tracing; not currently multi-turn)
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Triage prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior payments platform engineer acting as a triage assistant.
You have access to a knowledge base of past merchant issues with their root causes and resolutions.

Your job is to analyze an incoming merchant issue and produce a structured triage response.
Ground your analysis in the retrieved similar cases — don't speculate beyond what the evidence supports.

Guidelines:
- likely_root_cause: Be specific. Name the subsystem, config, or code path if the cases point to one.
- affected_component: Use the taxonomy: payment_processing, webhooks, subscription_billing, payout, fraud_risk, or other.
- recommended_action: Give the on-call engineer a concrete first step. "Check X" is better than "Investigate the system."
- confidence_score: 0.8+ means strong pattern match. 0.5–0.8 means plausible hypothesis. <0.5 means novel — say so.
- escalation_recommended: Set to true if the issue resembles a prior P0/P1 with >500 affected transactions.
- supporting_cases: Only cite cases that are genuinely relevant. 2–3 strong references beats 5 weak ones.
- caveats: Be honest about gaps. If the issue description is vague, note it.

Respond ONLY with valid JSON matching the TriageResponse schema. No prose before or after the JSON."""

USER_PROMPT_TEMPLATE = """
## Incoming Issue

{issue_description}

**Severity hint:** {severity}
**Merchant tier hint:** {merchant_tier}
**Component hint:** {component_hint}

## Retrieved Similar Cases from Knowledge Base

{retrieval_context}

---

Based on the above, produce a structured TriageResponse JSON.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: TriageState) -> dict:
    """
    Node 1: Retrieve semantically similar past cases from the vector store.

    Applies any available metadata filters from the incoming issue state
    before running the similarity search. Deduplicates results by issue_id
    so the LLM context contains distinct cases, not repeated chunks.
    """
    vector_store = load_vector_store()

    results = search(
        vector_store=vector_store,
        query=state["issue_description"],
        k=8,  # retrieve more than we need; dedup will trim it
        severity=state.get("severity"),
        merchant_tier=state.get("merchant_tier"),
        component=state.get("component_hint"),
    )

    # If filtered search returns < 3 results, fall back to unfiltered search
    # to avoid starving the LLM of context on narrow queries
    if len(results) < 3:
        logger.info(
            "Filtered retrieval returned %d results; falling back to unfiltered search.",
            len(results),
        )
        unfiltered = search(
            vector_store=vector_store,
            query=state["issue_description"],
            k=6,
        )
        results = results + unfiltered

    deduped = deduplicate_results(results)
    # Cap at top-5 unique issues for context
    top_cases = deduped[:5]
    context = format_results_for_context(top_cases)

    logger.info("Retrieved %d unique cases for triage.", len(top_cases))

    return {
        "retrieved_cases": top_cases,
        "retrieval_context": context,
        "messages": [HumanMessage(content=f"Retrieved {len(top_cases)} similar cases.")],
    }


def triage_node(state: TriageState) -> dict:
    """
    Node 2: Run the LLM triage chain over the retrieved context.

    Uses LCEL (LangChain Expression Language) composition:
        prompt | llm | json_parser

    The JsonOutputParser handles the structured output extraction. In
    production we use with_structured_output() on gpt-4o directly, but
    the manual parser approach here makes the prompt engineering more
    transparent and allows custom fallback logic.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,  # low temp for consistent, grounded responses
        openai_api_key=api_key,
    )

    # Build the prompt
    user_content = USER_PROMPT_TEMPLATE.format(
        issue_description=state["issue_description"],
        severity=state.get("severity") or "not specified",
        merchant_tier=state.get("merchant_tier") or "not specified",
        component_hint=state.get("component_hint") or "not specified",
        retrieval_context=state["retrieval_context"],
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    # LCEL chain: llm → JSON parser → Pydantic model
    json_parser = JsonOutputParser(pydantic_object=TriageResponse)

    response = llm.invoke(messages)
    raw_json = json_parser.parse(response.content)

    # Construct the TriageResponse from the parsed dict
    triage_response = TriageResponse(**raw_json)

    logger.info(
        "Triage complete: component=%s, confidence=%.2f, escalation=%s",
        triage_response.affected_component,
        triage_response.confidence_score,
        triage_response.escalation_recommended,
    )

    return {
        "triage_response": triage_response,
        "messages": [
            HumanMessage(
                content=(
                    f"Triage result: {triage_response.affected_component} "
                    f"(confidence={triage_response.confidence_score:.2f})"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_triage_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph for the triage agent.

    The graph is intentionally simple (linear: retrieve → triage → END)
    but structured as a graph to make it easy to extend — e.g., adding a
    human-in-the-loop confirmation node for P0 escalations, or a parallel
    node that queries an external incident management API.
    """
    graph = StateGraph(TriageState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("triage", triage_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "triage")
    graph.add_edge("triage", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_triage(
    issue_description: str,
    severity: Optional[str] = None,
    merchant_tier: Optional[str] = None,
    component_hint: Optional[str] = None,
) -> TriageResponse:
    """
    Main entry point for the triage agent.

    Args:
        issue_description: Free-text description of the incoming merchant issue.
        severity: Optional severity hint ("P0", "P1", "P2"). If provided,
                  retrieval will bias toward similar-severity past cases.
        merchant_tier: Optional tier hint ("enterprise", "mid-market", "smb").
        component_hint: Optional component hint to narrow retrieval.

    Returns:
        A TriageResponse with root cause, recommendation, confidence, and
        supporting cases from the knowledge base.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        RuntimeError: If the vector store is empty (run ingest.py first).
    """
    app = build_triage_graph()

    initial_state: TriageState = {
        "issue_description": issue_description,
        "severity": severity,
        "merchant_tier": merchant_tier,
        "component_hint": component_hint,
        "retrieved_cases": [],
        "retrieval_context": "",
        "triage_response": None,
        "messages": [],
    }

    final_state = app.invoke(initial_state)
    return final_state["triage_response"]


# ---------------------------------------------------------------------------
# CLI entry point for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    test_issue = (
        "Enterprise merchant is seeing a spike in payment declines starting about 20 minutes ago. "
        "Customers are getting generic 'card declined' errors on Visa and Mastercard. "
        "The merchant's own metrics show ~40% of checkout attempts failing. "
        "No code was deployed on our side in the last 6 hours. "
        "Issue appears to be across all card types."
    )

    result = run_triage(
        issue_description=test_issue,
        severity="P0",
        merchant_tier="enterprise",
    )

    print("\n=== TRIAGE RESULT ===")
    print(json.dumps(result.dict(), indent=2))
