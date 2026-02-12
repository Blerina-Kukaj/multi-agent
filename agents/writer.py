"""
Writer Agent – produces the structured deliverable from research notes.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from langchain_core.messages import SystemMessage, HumanMessage

import config
from agents.state import GraphState, AgentTrace, ActionItem
from agents.prompts import WRITER_SYSTEM, WRITER_USER, WRITER_SYSTEM_ANALYST, WRITER_USER_ANALYST
from agents.llm import get_llm

logger = logging.getLogger(__name__)


def _format_notes(state: GraphState) -> str:
    """Serialize research notes for the prompt."""
    notes = state.get("research_notes", [])
    parts: list[str] = []
    for i, n in enumerate(notes, 1):
        parts.append(f"Note {i}: {n.content}\n  Citation: {n.citation}")
    return "\n\n".join(parts)


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


def writer_node(state: GraphState) -> dict:
    """LangGraph node: produce the four-section deliverable."""
    trace = AgentTrace(agent="Writer")

    notes_text = _format_notes(state)
    if not notes_text:
        trace.finish(error="No research notes provided")
        traces = list(state.get("traces", []))
        traces.append(trace)
        return {"traces": traces}

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

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)

        action_items = [
            ActionItem(
                action=a["action"],
                owner=a.get("owner", "TBD"),
                due_date=a.get("due_date", "TBD"),
                confidence=a.get("confidence", "Medium"),
            )
            for a in parsed.get("action_items", [])
        ]

        tokens_in = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
        tokens_out = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
        trace.finish(tokens_in=tokens_in, tokens_out=tokens_out)
        logger.info("Writer produced deliverable")

        result = {
            "executive_summary": parsed.get("executive_summary", ""),
            "client_email": parsed.get("client_email", ""),
            "action_items": action_items,
            "sources_section": parsed.get("sources_section", ""),
        }

        # If no real evidence was found, drop the email entirely
        if not _has_real_evidence(state):
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

    except Exception as e:
        trace.finish(error=str(e))
        logger.error("Writer failed: %s", e)
        result = {}

    traces = list(state.get("traces", []))
    traces.append(trace)
    result["traces"] = traces

    return result
