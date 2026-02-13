"""
Research Agent – retrieves grounded evidence for each sub-question
from the vector store and produces research notes with citations.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage

import config
from agents.state import GraphState, ResearchNote, AgentMetrics
from agents.prompts import RESEARCHER_SYSTEM, RESEARCHER_USER
from agents.llm import get_llm
from agents.utils import parse_llm_json
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
        # Use content hash as fallback ID to avoid deduplicating different uncited chunks
        dedup_key = cid if cid else hash(doc.page_content)
        if dedup_key not in seen_ids:
            seen_ids.add(dedup_key)
            formatted_chunks.append(
                f"--- {cid} ---\n{doc.page_content}\n"
            )

    return "\n".join(formatted_chunks)


def researcher_node(state: GraphState) -> dict:
    """LangGraph node: produce research notes grounded in retrieved evidence."""
    plan = state.get("plan", [])
    if not plan:
        logger.error("Researcher: No plan provided")
        return {"research_notes": []}

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

    metrics = AgentMetrics(agent="researcher")
    t0 = time.perf_counter()

    try:
        response = llm.invoke(messages)
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        usage = getattr(response, "usage_metadata", None) or {}
        metrics.input_tokens = usage.get("input_tokens", 0)
        metrics.output_tokens = usage.get("output_tokens", 0)

        raw = response.content.strip()
        parsed = parse_llm_json(raw)

        notes: list[ResearchNote] = []
        for item in parsed:
            citations = item.get("citations", [])
            note = ResearchNote(
                content=f"{item['sub_question']}\n{item['note']}",
                citation="; ".join(citations) if citations else "No citation",
            )
            notes.append(note)

        logger.info("Researcher produced %d notes", len(notes))

    except json.JSONDecodeError as e:
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        metrics.error = f"JSON parse: {e}"
        logger.error("Researcher JSON parse failed: %s", e)
        notes = []

    except Exception as e:
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        metrics.error = str(e)
        logger.error("Researcher failed: %s", e)
        notes = []

    return {"research_notes": notes, "agent_metrics": [metrics]}
