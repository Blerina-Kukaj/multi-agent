"""
Shared LangGraph state schema for the multi-agent workflow.

All agents read from and write to this typed dictionary so that data
flows cleanly through the Plan → Research → Draft → Verify → Deliver pipeline.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict
from dataclasses import dataclass


# ── Research note produced by the Research Agent ───────────────────────────
@dataclass
class ResearchNote:
    """A single research finding with citation."""
    content: str
    citation: str          # e.g. "[fda_clinical_trial_guidelines_2025.txt | Chunk #3]"
    relevance_score: float = 0.0


# ── Action item in the final deliverable ───────────────────────────────────
@dataclass
class ActionItem:
    """One row in the action list."""
    action: str
    owner: str
    due_date: str
    confidence: str   # High / Medium / Low


# ── Per-agent observability metrics ────────────────────────────────────────
@dataclass
class AgentMetrics:
    """Latency, token usage, and error tracking for one agent."""
    agent: str               # "planner", "researcher", "writer", "verifier"
    latency_s: float = 0.0   # Wall-clock seconds
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""          # Empty = no error


# ── Central graph state ────────────────────────────────────────────────────
class GraphState(TypedDict, total=False):
    # Input
    task: str                            # Raw user request
    goal: str                            # Optional goal context
    output_mode: str                     # "executive" or "analyst"

    # Planner output
    plan: list[str]                      # Ordered sub-questions / steps

    # Research output
    research_notes: list[ResearchNote]   # Grounded notes with citations

    # Writer output
    executive_summary: str               # ≤ 150 words
    client_email: str                    # Client-ready email body
    action_items: list[ActionItem]       # Owner + due date + confidence
    sources_section: str                 # Formatted citations

    # Verifier output
    verification_passed: bool
    verification_issues: list[str]       # Issues found by verifier
    verified_summary: str                # Summary after verification pass
    verified_email: str                  # Email after verification pass
    verified_action_items: list[ActionItem]
    verified_sources: str

    # Observability
    agent_metrics: Annotated[list[AgentMetrics], operator.add]  # Accumulated per-agent
