"""
Writer Agent – produces the structured deliverable from research notes.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import GraphState, ActionItem, AgentMetrics
from agents.prompts import WRITER_SYSTEM, WRITER_USER, WRITER_SYSTEM_ANALYST, WRITER_USER_ANALYST
from agents.llm import get_llm
from agents.utils import parse_llm_json, has_real_evidence

logger = logging.getLogger(__name__)


def _format_notes(state: GraphState) -> str:
    """Serialize research notes for the prompt."""
    notes = state.get("research_notes", [])
    parts: list[str] = []
    for i, n in enumerate(notes, 1):
        parts.append(f"Note {i}: {n.content}\n  Citation: {n.citation}")
    return "\n\n".join(parts)


def writer_node(state: GraphState) -> dict:
    """LangGraph node: produce the four-section deliverable."""
    notes = state.get("research_notes", [])

    # If no notes or none carry real evidence, return a clean "out of scope"
    # response immediately — no LLM call needed.
    if not notes or not has_real_evidence(notes):
        logger.info("Writer: No grounded evidence — returning out-of-scope response")
        return {
            "executive_summary": "",
            "client_email": "",
            "action_items": [],
            "sources_section": (
                "No sources available — the query did not match any indexed documents."
            ),
            "agent_metrics": [AgentMetrics(agent="writer")],
        }

    notes_text = _format_notes(state)

    llm = get_llm(temperature=0.3, max_tokens=1500)

    # Select prompts based on output mode
    mode = state.get("output_mode", "executive")
    if mode == "analyst":
        sys_prompt = WRITER_SYSTEM_ANALYST
        usr_prompt = WRITER_USER_ANALYST
    else:
        sys_prompt = WRITER_SYSTEM
        usr_prompt = WRITER_USER

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(
            content=usr_prompt.format(
                task=state.get("task", ""),
                goal=state.get("goal", ""),
                today=date.today().strftime("%B %d, %Y"),
                research_notes=notes_text,
            )
        ),
    ]

    metrics = AgentMetrics(agent="writer")
    t0 = time.perf_counter()

    try:
        response = llm.invoke(messages)
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        usage = getattr(response, "usage_metadata", None) or {}
        metrics.input_tokens = usage.get("input_tokens", 0)
        metrics.output_tokens = usage.get("output_tokens", 0)

        raw = response.content.strip()
        parsed = parse_llm_json(raw)

        action_items = [
            ActionItem(
                action=a["action"],
                owner=a.get("owner", "TBD"),
                due_date=a.get("due_date", "TBD"),
                confidence=a.get("confidence", "Medium"),
            )
            for a in parsed.get("action_items", [])
        ]

        logger.info("Writer produced deliverable")

        result = {
            "executive_summary": parsed.get("executive_summary", ""),
            "client_email": parsed.get("client_email", ""),
            "action_items": action_items,
            "sources_section": parsed.get("sources_section", ""),
        }

        # If no real evidence was found, drop the email entirely
        if not has_real_evidence(state.get("research_notes", [])):
            result["client_email"] = ""

        # Normalize sources_section to string if LLM returned a list or dict
        src = result["sources_section"]
        if isinstance(src, list):
            result["sources_section"] = "\n".join(str(s) for s in src)
        elif isinstance(src, dict):
            result["sources_section"] = "\n".join(
                f"- {k}: {v}" for k, v in src.items()
            )
        elif not isinstance(src, str):
            result["sources_section"] = str(src)

    except json.JSONDecodeError as e:
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        metrics.error = f"JSON parse: {e}"
        logger.error("Writer JSON parse failed: %s", e)
        result = {}

    except Exception as e:
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        metrics.error = str(e)
        logger.error("Writer failed: %s", e)
        result = {}

    result["agent_metrics"] = [metrics]
    return result
