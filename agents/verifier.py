"""
Verifier Agent – audits the Writer's deliverable for hallucinations,
missing evidence, contradictions, and citation accuracy.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

import config
from agents.state import GraphState, AgentTrace, ActionItem
from agents.prompts import VERIFIER_SYSTEM, VERIFIER_USER
from agents.llm import get_llm

logger = logging.getLogger(__name__)


def _has_real_evidence(state: GraphState) -> bool:
    """Return True only if at least one research note has real sourced content."""
    notes = state.get("research_notes", [])
    if not notes:
        return False
    for n in notes:
        text = n.content.lower()
        if "not found in sources" not in text and "additional information needed" not in text:
            return True
    return False


def _format_notes_for_verifier(state: GraphState) -> str:
    """Serialize research notes for verifier context (compact format)."""
    notes = state.get("research_notes", [])
    parts: list[str] = []
    for i, n in enumerate(notes, 1):
        # Truncate long notes to first 200 chars to reduce prompt size
        content = n.content[:200] + "..." if len(n.content) > 200 else n.content
        parts.append(f"Note {i}: {content} | Citation: {n.citation}")
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
    trace = AgentTrace(agent="Verifier")

    llm = get_llm(temperature=0.0, max_tokens=1500)

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
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # Extract the first complete JSON object using brace-counting
        if "{" in raw:
            start = raw.index("{")
            depth = 0
            end = start
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            raw = raw[start:end + 1]
        parsed = json.loads(raw)

        verified_action_items = [
            ActionItem(
                action=a["action"],
                owner=a.get("owner", "TBD"),
                due_date=a.get("due_date", "TBD"),
                confidence=a.get("confidence", "Medium"),
            )
            for a in parsed.get("verified_action_items", [])
        ]

        tokens_in = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
        tokens_out = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
        trace.finish(tokens_in=tokens_in, tokens_out=tokens_out)

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
        if not _has_real_evidence(state):
            result["verified_email"] = ""

    except Exception as e:
        trace.finish(error=str(e))
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

    traces = list(state.get("traces", []))
    traces.append(trace)
    result["traces"] = traces

    return result
