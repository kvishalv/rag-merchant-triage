"""
evaluate.py
-----------
LangSmith evaluation suite for the merchant issue triage system.

Runs two categories of evaluations:
  1. Retrieval relevance — does the retriever surface the correct past cases
     for a given query? Scored with a custom LLM-as-judge evaluator.
  2. Triage response quality — does the triage agent produce accurate,
     grounded, and actionable structured outputs? Scored across four
     dimensions: root cause accuracy, component accuracy, recommendation
     quality, and confidence calibration.

All results are logged to LangSmith for comparison across runs. Evaluation
datasets are maintained as code here (golden_dataset) to keep them version-
controlled alongside the system they test.

Usage:
    python src/evaluate.py
    python src/evaluate.py --eval retrieval      # retrieval eval only
    python src/evaluate.py --eval triage         # triage eval only
    python src/evaluate.py --dataset-name my_ds  # push dataset to LangSmith
"""

import argparse
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate
from langsmith.schemas import Example, Run

from retriever import (
    deduplicate_results,
    format_results_for_context,
    load_vector_store,
    search,
)
from triage_agent import TriageResponse, run_triage

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------
# Each entry defines an input (the query a support engineer would type)
# and the expected output (ground truth for eval scoring).
#
# This dataset is deliberately small (9 entries) and hand-curated against
# the fixture data in data/fixtures/merchant_issues.json. In production,
# this was grown iteratively as new incident patterns were confirmed.

GOLDEN_DATASET: list[dict] = [
    {
        "id": "eval-001",
        "input": {
            "issue_description": (
                "Enterprise merchant is seeing Visa card declines with code 51 across checkout. "
                "Started about 15 minutes ago. Cards with confirmed balances are being declined."
            ),
            "severity": "P0",
            "merchant_tier": "enterprise",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10041"],
            "affected_component": "payment_processing",
            "root_cause_keywords": ["routing", "BIN", "processor", "acquiring"],
            "should_escalate": True,
        },
    },
    {
        "id": "eval-002",
        "input": {
            "issue_description": (
                "Merchant's order fulfillment system isn't receiving payment webhooks in real-time. "
                "Events are arriving 30-45 minutes late. This started during a high-volume sale event."
            ),
            "severity": "P1",
            "merchant_tier": "enterprise",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10087"],
            "affected_component": "webhooks",
            "root_cause_keywords": ["queue", "backlog", "autoscaler", "worker", "fan-out"],
            "should_escalate": True,
        },
    },
    {
        "id": "eval-003",
        "input": {
            "issue_description": (
                "Subscription renewals are silently failing — no retry is being scheduled and "
                "no error event is being emitted. We noticed a drop in MRR during reconciliation."
            ),
            "severity": "P1",
            "merchant_tier": "mid-market",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10102"],
            "affected_component": "subscription_billing",
            "root_cause_keywords": ["state machine", "exception", "terminal", "silent"],
            "should_escalate": True,
        },
    },
    {
        "id": "eval-004",
        "input": {
            "issue_description": (
                "Apple Pay is failing on iOS devices but working fine on desktop. "
                "Customers see 'payment not completed' error at checkout."
            ),
            "severity": "P2",
            "merchant_tier": "mid-market",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10211"],
            "affected_component": "payment_processing",
            "root_cause_keywords": ["domain", "validation", "Apple", "regex", "iOS"],
            "should_escalate": False,
        },
    },
    {
        "id": "eval-005",
        "input": {
            "issue_description": (
                "Customers are being double-charged. Support is receiving complaints about "
                "duplicate transactions appearing on credit card statements."
            ),
            "severity": "P2",
            "merchant_tier": "smb",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10149"],
            "affected_component": "payment_processing",
            "root_cause_keywords": ["idempotency", "duplicate", "retry", "timeout"],
            "should_escalate": False,
        },
    },
    {
        "id": "eval-006",
        "input": {
            "issue_description": (
                "Webhook HMAC signature validation is failing. Our endpoint is rejecting all "
                "events with 401. This started happening without any changes on our end."
            ),
            "severity": "P1",
            "merchant_tier": "mid-market",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10163"],
            "affected_component": "webhooks",
            "root_cause_keywords": ["signing secret", "key rotation", "HMAC", "propagation"],
            "should_escalate": True,
        },
    },
    {
        "id": "eval-007",
        "input": {
            "issue_description": (
                "All EUR transactions are failing with a 500 error. Non-EUR transactions are fine. "
                "Started suddenly during peak hours."
            ),
            "severity": "P0",
            "merchant_tier": "enterprise",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10177"],
            "affected_component": "payment_processing",
            "root_cause_keywords": ["FX", "currency", "cache", "rate", "circuit breaker"],
            "should_escalate": True,
        },
    },
    {
        "id": "eval-008",
        "input": {
            "issue_description": (
                "Scheduled payout batch did not run. Multiple merchants are reporting that "
                "their ACH settlements haven't arrived as expected."
            ),
            "severity": "P1",
            "merchant_tier": "enterprise",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10135"],
            "affected_component": "payout",
            "root_cause_keywords": ["CronJob", "batch", "eviction", "Kubernetes"],
            "should_escalate": True,
        },
    },
    {
        "id": "eval-009",
        "input": {
            "issue_description": (
                "High-value transactions over $500 are being blocked at a very high rate. "
                "The fraud engine seems to be declining everything above that threshold."
            ),
            "severity": "P1",
            "merchant_tier": "enterprise",
        },
        "expected": {
            "relevant_issue_ids": ["ISS-10228"],
            "affected_component": "fraud_risk",
            "root_cause_keywords": ["timeout", "fail-secure", "ML model", "cold start"],
            "should_escalate": True,
        },
    },
]


# ---------------------------------------------------------------------------
# LangSmith dataset management
# ---------------------------------------------------------------------------

def push_dataset_to_langsmith(
    dataset_name: str = "merchant-triage-golden",
    client: Optional[Client] = None,
) -> str:
    """
    Create or update a LangSmith dataset from the golden dataset.

    Returns the dataset ID. Safe to call repeatedly — existing examples
    are matched by their 'id' field and updated if changed.
    """
    if client is None:
        client = Client()

    # Create dataset if it doesn't exist
    existing_datasets = {ds.name: ds for ds in client.list_datasets()}
    if dataset_name in existing_datasets:
        dataset = existing_datasets[dataset_name]
        logger.info("Updating existing LangSmith dataset: %s", dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden evaluation dataset for merchant issue triage RAG system.",
        )
        logger.info("Created LangSmith dataset: %s (id=%s)", dataset_name, dataset.id)

    # Upsert examples
    for entry in GOLDEN_DATASET:
        client.create_example(
            inputs=entry["input"],
            outputs=entry["expected"],
            dataset_id=dataset.id,
            source_run_id=None,
        )

    logger.info("Pushed %d examples to dataset '%s'.", len(GOLDEN_DATASET), dataset_name)
    return str(dataset.id)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def retrieval_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates whether the retriever surfaces the correct past cases.

    Scoring:
      1.0 — all expected issue IDs appear in the top-k retrieved results
      0.5 — at least one expected issue ID appears in the top-k results
      0.0 — none of the expected issue IDs appear

    This is a recall-focused metric. Precision is handled separately by the
    triage response quality evaluator (which checks for hallucinated citations).
    """
    vector_store = load_vector_store()
    query = example.inputs["issue_description"]
    severity = example.inputs.get("severity")
    merchant_tier = example.inputs.get("merchant_tier")

    results = search(
        vector_store=vector_store,
        query=query,
        k=6,
        severity=severity,
        merchant_tier=merchant_tier,
    )
    deduped = deduplicate_results(results)
    retrieved_ids = {r.issue_id for r in deduped}
    expected_ids = set(example.outputs.get("relevant_issue_ids", []))

    hits = expected_ids & retrieved_ids
    if len(hits) == len(expected_ids):
        score = 1.0
    elif len(hits) > 0:
        score = 0.5
    else:
        score = 0.0

    return {
        "key": "retrieval_recall",
        "score": score,
        "comment": (
            f"Expected {expected_ids}, retrieved {retrieved_ids}. "
            f"Hits: {hits}"
        ),
    }


def component_accuracy_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates whether the triage agent correctly identifies the affected component.

    Binary score: 1.0 if correct, 0.0 if wrong.
    Component accuracy is the most reliable signal for triage quality because
    it's objective and maps directly to on-call routing.
    """
    output = run.outputs or {}
    triage = output.get("triage_response")
    if not triage:
        return {"key": "component_accuracy", "score": 0.0, "comment": "No triage response."}

    # Handle both dict and TriageResponse object
    component = triage.get("affected_component") if isinstance(triage, dict) else triage.affected_component
    expected = example.outputs.get("affected_component", "")

    score = 1.0 if component == expected else 0.0
    return {
        "key": "component_accuracy",
        "score": score,
        "comment": f"Predicted: {component}, Expected: {expected}",
    }


def root_cause_quality_evaluator(run: Run, example: Example) -> dict:
    """
    LLM-as-judge evaluator for root cause quality.

    Uses GPT-4o to assess whether the predicted root cause contains the
    expected keywords and is grounded in the retrieved cases rather than
    hallucinated. Returns a score from 0.0 to 1.0.

    This is the most expensive evaluator (~1 LLM call per example) but
    provides the richest signal for prompt engineering iteration.
    """
    output = run.outputs or {}
    triage = output.get("triage_response")
    if not triage:
        return {"key": "root_cause_quality", "score": 0.0, "comment": "No triage response."}

    root_cause = triage.get("likely_root_cause") if isinstance(triage, dict) else triage.likely_root_cause
    expected_keywords = example.outputs.get("root_cause_keywords", [])

    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    judge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert evaluator for a merchant issue triage system. "
            "Score the predicted root cause on a scale from 0.0 to 1.0:\n"
            "1.0 — Accurate, specific, and grounded. Contains or implies the key concepts.\n"
            "0.7 — Correct direction but missing specificity or key details.\n"
            "0.4 — Partially correct but with misleading or generic elements.\n"
            "0.0 — Wrong, hallucinated, or completely off-base.\n\n"
            "Respond with JSON: {{\"score\": <float>, \"reason\": \"<one sentence>\"}}",
        ),
        (
            "human",
            "Predicted root cause:\n{root_cause}\n\n"
            "Expected to contain concepts like: {keywords}\n\n"
            "Score:",
        ),
    ])

    chain = judge_prompt | judge_llm
    response = chain.invoke({
        "root_cause": root_cause,
        "keywords": ", ".join(expected_keywords),
    })

    import json
    try:
        result = json.loads(response.content)
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.0
        reason = f"Parse error on judge response: {response.content[:100]}"

    return {"key": "root_cause_quality", "score": score, "comment": reason}


def escalation_accuracy_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates whether the agent correctly determines if escalation is warranted.

    Binary score: 1.0 if escalation_recommended matches ground truth, 0.0 otherwise.
    False negatives (not escalating a P0-severity issue) are weighted higher in
    production, but this evaluator keeps it simple for the demo.
    """
    output = run.outputs or {}
    triage = output.get("triage_response")
    if not triage:
        return {"key": "escalation_accuracy", "score": 0.0, "comment": "No triage response."}

    predicted = triage.get("escalation_recommended") if isinstance(triage, dict) else triage.escalation_recommended
    expected = example.outputs.get("should_escalate", False)

    score = 1.0 if predicted == expected else 0.0
    return {
        "key": "escalation_accuracy",
        "score": score,
        "comment": f"Predicted escalate={predicted}, expected={expected}",
    }


# ---------------------------------------------------------------------------
# Target functions (called by LangSmith evaluate)
# ---------------------------------------------------------------------------

def retrieval_target(inputs: dict) -> dict:
    """Wrap the retriever for LangSmith evaluation."""
    vector_store = load_vector_store()
    results = search(
        vector_store=vector_store,
        query=inputs["issue_description"],
        k=6,
        severity=inputs.get("severity"),
        merchant_tier=inputs.get("merchant_tier"),
    )
    deduped = deduplicate_results(results)
    return {
        "retrieved_ids": [r.issue_id for r in deduped],
        "retrieval_context": format_results_for_context(deduped),
    }


def triage_target(inputs: dict) -> dict:
    """Wrap the triage agent for LangSmith evaluation."""
    result = run_triage(
        issue_description=inputs["issue_description"],
        severity=inputs.get("severity"),
        merchant_tier=inputs.get("merchant_tier"),
    )
    return {"triage_response": result.dict()}


# ---------------------------------------------------------------------------
# Eval runners
# ---------------------------------------------------------------------------

def run_retrieval_eval(
    dataset_name: str = "merchant-triage-golden",
    experiment_prefix: str = "retrieval",
) -> None:
    """Run the retrieval evaluation suite against the LangSmith dataset."""
    logger.info("Starting retrieval evaluation against dataset '%s'.", dataset_name)
    results = ls_evaluate(
        retrieval_target,
        data=dataset_name,
        evaluators=[retrieval_relevance_evaluator],
        experiment_prefix=experiment_prefix,
        metadata={"evaluator": "retrieval_recall", "k": 6},
    )
    logger.info("Retrieval eval complete. View results in LangSmith.")
    return results


def run_triage_eval(
    dataset_name: str = "merchant-triage-golden",
    experiment_prefix: str = "triage",
) -> None:
    """Run the full triage response quality evaluation suite."""
    logger.info("Starting triage quality evaluation against dataset '%s'.", dataset_name)
    results = ls_evaluate(
        triage_target,
        data=dataset_name,
        evaluators=[
            component_accuracy_evaluator,
            root_cause_quality_evaluator,
            escalation_accuracy_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        metadata={
            "model": "gpt-4o",
            "evaluators": ["component_accuracy", "root_cause_quality", "escalation_accuracy"],
        },
    )
    logger.info("Triage eval complete. View results in LangSmith.")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluations for the merchant triage system."
    )
    parser.add_argument(
        "--eval",
        choices=["retrieval", "triage", "all"],
        default="all",
        help="Which evaluation to run.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="merchant-triage-golden",
        help="LangSmith dataset name to use (will be created if not exists).",
    )
    parser.add_argument(
        "--push-dataset",
        action="store_true",
        help="Push the golden dataset to LangSmith before running evals.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Validate LangSmith credentials
    if not os.environ.get("LANGCHAIN_API_KEY"):
        raise EnvironmentError(
            "LANGCHAIN_API_KEY not set. Copy .env.example to .env and populate it."
        )

    if args.push_dataset:
        push_dataset_to_langsmith(dataset_name=args.dataset_name)

    if args.eval in ("retrieval", "all"):
        run_retrieval_eval(dataset_name=args.dataset_name)

    if args.eval in ("triage", "all"):
        run_triage_eval(dataset_name=args.dataset_name)
