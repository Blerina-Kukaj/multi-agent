"""
LangGraph workflow – wires the four agents into a linear
Plan → Research → Draft → Verify → Deliver pipeline.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import StateGraph, END

from agents.state import GraphState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.verifier import verifier_node


def _should_verify(state: GraphState) -> str:
    """Route to verifier only if the writer produced output."""
    if state.get("executive_summary") or state.get("client_email"):
        return "verifier"
    return END


@lru_cache(maxsize=1)
def build_graph() -> StateGraph:
    """Construct and compile the multi-agent LangGraph (cached after first call)."""
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("verifier", verifier_node)

    # Pipeline: Plan → Research → Draft → (Verify if output exists) → END
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_conditional_edges("writer", _should_verify, {"verifier": "verifier", END: END})
    workflow.add_edge("verifier", END)

    return workflow.compile()


def run_pipeline(task: str, goal: str = "", output_mode: str = "executive") -> GraphState:
    """
    Execute the full multi-agent pipeline and return the final state.

    Parameters
    ----------
    task : str
        The user's business request / question.
    goal : str, optional
        Additional context about the desired outcome.
    output_mode : str
        "executive" for concise C-suite output, "analyst" for detailed data-rich output.

    Returns
    -------
    GraphState
        The final state containing all deliverables and metadata.
    """
    graph = build_graph()
    initial_state: GraphState = {
        "task": task,
        "goal": goal or task,
        "output_mode": output_mode,
    }
    final_state = graph.invoke(initial_state)
    return final_state


# Quick smoke test
if __name__ == "__main__":
    import json

    result = run_pipeline(
        task="What are the key FDA guidelines affecting oncology clinical trial timelines?",
        goal="Produce recommendations for accelerating Phase III oncology trials.",
    )
    print("=== EXECUTIVE SUMMARY ===")
    print(result.get("verified_summary", result.get("executive_summary", "N/A")))
    print("\n=== ISSUES ===")
    for issue in result.get("verification_issues", []):
        print(f"  • {issue}")
