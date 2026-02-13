"""
Planner Agent – decomposes the user's task into sub-questions.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import GraphState, AgentMetrics
from agents.prompts import PLANNER_SYSTEM, PLANNER_USER
from agents.llm import get_llm
from agents.utils import parse_llm_json

logger = logging.getLogger(__name__)


def planner_node(state: GraphState) -> dict:
    """LangGraph node: decompose the task into a plan (list of sub-questions)."""
    task = state.get("task", "")
    goal = state.get("goal", task)

    llm = get_llm(temperature=0.2, max_tokens=300)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=PLANNER_USER.format(task=task, goal=goal)),
    ]

    metrics = AgentMetrics(agent="planner")
    t0 = time.perf_counter()

    try:
        response = llm.invoke(messages)
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        usage = getattr(response, "usage_metadata", None) or {}
        metrics.input_tokens = usage.get("input_tokens", 0)
        metrics.output_tokens = usage.get("output_tokens", 0)

        raw = response.content.strip()
        plan: list[str] = parse_llm_json(raw)

        logger.info("Planner created %d sub-questions", len(plan))

    except json.JSONDecodeError as e:
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        metrics.error = f"JSON parse: {e}"
        logger.error("Planner JSON parse failed: %s", e)
        plan = [task]

    except Exception as e:
        metrics.latency_s = round(time.perf_counter() - t0, 2)
        metrics.error = str(e)
        logger.error("Planner failed: %s", e)
        plan = [task]

    return {"plan": plan, "agent_metrics": [metrics]}
