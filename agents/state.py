"""
Shared LangGraph state schema for the multi-agent workflow.

All agents read from and write to this typed dictionary so that data
flows cleanly through the Plan → Research → Draft → Verify → Deliver pipeline.
"""

from __future__ import annotations

import time
from typing import TypedDict
from dataclasses import dataclass, field


# ── Observability record for a single agent step ───────────────────────────
@dataclass
class AgentTrace:
    """One row in the observability table."""
    agent: str = ""
    status: str = "pending"       # pending | running | success | error
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)

    def finish(self, tokens_in: int = 0, tokens_out: int = 0, error: str = "") -> None:
        self.latency_s = round(time.time() - self.started_at, 2)
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.status = "error" if error else "success"
        self.error = error


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
    traces: list[AgentTrace]
