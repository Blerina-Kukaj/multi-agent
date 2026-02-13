"""
Verifier Agent – audits the Writer's deliverable for hallucinations,
missing evidence, contradictions, and citation accuracy.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import GraphState, ActionItem, AgentMetrics
from agents.prompts import VERIFIER_SYSTEM, VERIFIER_USER
from agents.llm import get_llm
from agents.utils import parse_llm_json, has_real_evidence

logger = logging.getLogger(__name__)


def _format_notes_for_verifier(state: GraphState) -> str:
    """Serialize research notes for verifier context."""
    notes = state.get("research_notes", [])
    parts: list[str] = []
    for i, n in enumerate(notes, 1):
        parts.append(f"Note {i}: {n.content} | Citation: {n.citation}")
    return "\n".join(parts)


def _format_action_items(items: list[ActionItem]) -> str:
    """Serialize action items as JSON string for prompt."""
    return json.dumps(
        [
            {
                "action": a.action,
                "owner": a.owner,
                "due_date": a.due_date,
                "confidence": a.confidence,
            }
            for a in items
        ],
        indent=2,
    )


def verifier_node(state: GraphState) -> dict:
    """LangGraph node: verify the writer's deliverable."""
    # If no real evidence, pass through writer output (skip LLM verification)
    if not has_real_evidence(state.get("research_notes", [])):
        logger.info("Verifier: No real evidence — passing through writer output")
        return {
            "verification_passed": True,
            "verification_issues": [],
            "verified_summary": state.get("executive_summary", ""),
            "verified_email": state.get("client_email", ""),
            "verified_action_items": state.get("action_items", []),
            "verified_sources": state.get("sources_section", ""),
            "agent_metrics": [AgentMetrics(agent="verifier")],
        }

    metrics = AgentMetrics(agent="verifier")
    t0 = time.perf_counter()

    llm = get_llm(temperature=0.0, max_tokens=2000)

    messages = [
        SystemMessage(content=VERIFIER_SYSTEM),
        HumanMessage(
            content=VERIFIER_USER.format(
                task=state.get("task", ""),
                research_notes=_format_notes_for_verifier(state),
                executive_summary=state.get("executive_summary", ""),
                client_email=state.get("client_email", ""),
                action_items=_format_action_items(state.get("action_items", [])),
                sources_section=str(state.get("sources_section", "")),
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        usage = getattr(response, "usage_metadata", None) or {}
        metrics.input_tokens = usage.get("input_tokens", 0)
        metrics.output_tokens = usage.get("output_tokens", 0)

        raw = response.content.strip()
        parsed = parse_llm_json(raw)

        verified_action_items = [
            ActionItem(
                action=a["action"],
                owner=a.get("owner", "TBD"),
                due_date=a.get("due_date", "TBD"),
                confidence=a.get("confidence", "Medium"),
            )
            for a in parsed.get("verified_action_items", [])
        ]

        issues = parsed.get("issues", [])
        passed = parsed.get("verification_passed", len(issues) == 0)
        logger.info(
            "Verifier: passed=%s, issues=%d", passed, len(issues)
        )

        result = {
            "verification_passed": passed,
            "verification_issues": issues,
            "verified_summary": parsed.get("verified_summary", state.get("executive_summary", "")),
            "verified_email": parsed.get("verified_email", state.get("client_email", "")),
            "verified_action_items": verified_action_items or state.get("action_items", []),
            "verified_sources": parsed.get("verified_sources", state.get("sources_section", "")),
        }

        # If no real evidence exists, drop the email
        if not has_real_evidence(state.get("research_notes", [])):
            result["verified_email"] = ""

    except Exception as e:
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        metrics.error = str(e)
        logger.error("Verifier failed: %s", e)
        # On failure, pass through writer output unchanged
        result = {
            "verification_passed": False,
            "verification_issues": [f"Verifier error: {e}"],
            "verified_summary": state.get("executive_summary", ""),
            "verified_email": state.get("client_email", ""),
            "verified_action_items": state.get("action_items", []),
            "verified_sources": state.get("sources_section", ""),
        }

    result["agent_metrics"] = [metrics]
    return result
