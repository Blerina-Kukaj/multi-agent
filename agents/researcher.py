"""
Research Agent – retrieves grounded evidence for each sub-question
from the vector store and produces research notes with citations.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

import config
from agents.state import GraphState, AgentTrace, ResearchNote
from agents.prompts import RESEARCHER_SYSTEM, RESEARCHER_USER
from agents.llm import get_llm
from retrieval.vector_store import retrieve

logger = logging.getLogger(__name__)


def _gather_chunks(plan: list[str], top_k: int = 8) -> str:
    """Retrieve relevant chunks using a single combined query.

    Combines all sub-questions into one query for a single embedding + retrieval
    call instead of N separate calls.
    """
    seen_ids: set[str] = set()
    formatted_chunks: list[str] = []

    # Single combined query captures all evidence in one retrieval call
    combined_query = " ".join(plan)
    docs = retrieve(combined_query, top_k=top_k)
    for doc in docs:
        cid = doc.metadata.get("citation", "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            formatted_chunks.append(
                f"--- {cid} ---\n{doc.page_content}\n"
            )

    return "\n".join(formatted_chunks)


def researcher_node(state: GraphState) -> dict:
    """LangGraph node: produce research notes grounded in retrieved evidence."""
    trace = AgentTrace(agent="Researcher")

    plan = state.get("plan", [])
    if not plan:
        trace.finish(error="No plan provided")
        traces = list(state.get("traces", []))
        traces.append(trace)
        return {"research_notes": [], "traces": traces}

    # Retrieve evidence
    chunks_text = _gather_chunks(plan, top_k=config.TOP_K)

    llm = get_llm(temperature=0.1, max_tokens=1500)

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(
            content=RESEARCHER_USER.format(
                sub_questions=json.dumps(plan, indent=2),
                chunks=chunks_text,
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # Fix common LLM JSON issues: trailing commas before } or ]
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        parsed = json.loads(raw)

        notes: list[ResearchNote] = []
        for item in parsed:
            citations = item.get("citations", [])
            note = ResearchNote(
                content=f"{item['sub_question']}\n{item['note']}",
                citation="; ".join(citations) if citations else "No citation",
            )
            notes.append(note)

        tokens_in = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
        tokens_out = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
        trace.finish(tokens_in=tokens_in, tokens_out=tokens_out)
        logger.info("Researcher produced %d notes", len(notes))

    except Exception as e:
        trace.finish(error=str(e))
        logger.error("Researcher failed: %s", e)
        notes = []

    traces = list(state.get("traces", []))
    traces.append(trace)

    return {"research_notes": notes, "traces": traces}
