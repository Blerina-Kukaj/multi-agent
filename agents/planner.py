"""
Planner Agent – decomposes the user's task into sub-questions.
"""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import config
from agents.state import GraphState, AgentTrace
from agents.prompts import PLANNER_SYSTEM, PLANNER_USER

logger = logging.getLogger(__name__)


def planner_node(state: GraphState) -> dict:
    """LangGraph node: decompose the task into a plan (list of sub-questions)."""
    trace = AgentTrace(agent="Planner")

    task = state.get("task", "")
    goal = state.get("goal", task)

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.2,
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=PLANNER_USER.format(task=task, goal=goal)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        plan: list[str] = json.loads(raw)

        tokens_in = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
        tokens_out = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
        trace.finish(tokens_in=tokens_in, tokens_out=tokens_out)
        logger.info("Planner created %d sub-questions", len(plan))

    except Exception as e:
        trace.finish(error=str(e))
        logger.error("Planner failed: %s", e)
        plan = [task]  # Fallback: use original task as single question

    traces = list(state.get("traces", []))
    traces.append(trace)

    return {"plan": plan, "traces": traces}
