"""
Streamlit UI for the Enterprise Multi-Agent Copilot.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import os
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from agents.graph import run_pipeline
from agents.state import ActionItem, AgentMetrics
from agents.guardrails import validate_and_sanitize, PromptInjectionError

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Multi-Agent Copilot",
    layout="wide",
)

st.title("Enterprise Multi-Agent Copilot")
st.caption("Healthcare & Life Sciences — Plan → Research → Draft → Verify → Deliver")

# ─── Session state ──────────────────────────────────────────────────────────
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# ─── Sample evaluation queries ─────────────────────────────────────────────
SAMPLE_QUERIES = [
    {
        "label": "1. FDA Adaptive Trial Guidelines",
        "task": "Analyze the impact of recent FDA adaptive trial design guidelines on Phase III oncology trial timelines and recommend process improvements.",
        "goal": "Produce actionable recommendations for the clinical operations leadership team.",
    },
    {
        "label": "2. Decentralized Clinical Trials",
        "task": "What are the best practices for implementing decentralized clinical trials in oncology, and what cost savings can be expected?",
        "goal": "Build a business case for DCT adoption in our oncology portfolio.",
    },
    {
        "label": "3. Real-World Evidence for Label Expansion",
        "task": "How can real-world evidence be used to support regulatory submissions for label expansion of approved oncology drugs?",
        "goal": "Advise the regulatory affairs team on RWE strategy for a post-approval submission.",
    },
    {
        "label": "4. AI/ML in Drug Discovery",
        "task": "Evaluate the current state of AI and machine learning applications in drug discovery and recommend investment priorities.",
        "goal": "Inform R&D leadership on where to allocate AI/ML investment for maximum impact.",
    },
    {
        "label": "5. Patient Data Privacy (Multi-Site Trial)",
        "task": "What patient data privacy safeguards are required for a multi-site international clinical trial using decentralized elements?",
        "goal": "Ensure compliance for a planned 15-country Phase III trial with telemedicine components.",
    },
    {
        "label": "6. Pharmacovigilance for Oncology Biologic",
        "task": "Assess pharmacovigilance requirements for a newly approved oncology biologic and recommend a post-market safety monitoring plan.",
        "goal": "Prepare the safety team for launch readiness.",
    },
    {
        "label": "7. Cold Chain Supply Resilience",
        "task": "How can healthcare organizations build supply chain resilience for temperature-sensitive biologic drugs?",
        "goal": "Develop a cold chain risk mitigation strategy for our biologics portfolio.",
    },
    {
        "label": "8. Value-Based Care Transition",
        "task": "What strategies should a mid-size health system use to transition from fee-for-service to value-based care models?",
        "goal": "Develop a 3-year transition roadmap for the executive team.",
    },
    {
        "label": "9. Clinical Ops Benchmarking",
        "task": "Compare the clinical operations efficiency of top-quartile pharmaceutical sponsors versus the industry average and identify key differentiators.",
        "goal": "Benchmark our clinical ops against best-in-class and identify improvement opportunities.",
    },
    {
        "label": "10. AI in Pharmacovigilance",
        "task": "What are the regulatory and operational considerations for using AI-assisted signal detection in pharmacovigilance?",
        "goal": "Evaluate whether to implement AI-based PV automation for our safety database.",
    },
]

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    output_mode = st.radio(
        "Output Mode",
        ["Executive", "Analyst"],
        index=0,
        help="**Executive**: Concise, C-suite oriented. **Analyst**: Detailed, data-rich analysis.",
    )

    st.divider()
    st.markdown("**Sample Queries**")
    sample_choice = st.selectbox(
        "Pick a test query",
        options=["-- Select --"] + [q["label"] for q in SAMPLE_QUERIES],
        index=0,
        label_visibility="collapsed",
    )
    if sample_choice != "-- Select --":
        selected = next(q for q in SAMPLE_QUERIES if q["label"] == sample_choice)
        st.session_state.selected_sample = selected
    else:
        st.session_state.selected_sample = None

    st.divider()
    st.markdown(
        "**Workflow:**\n"
        "1. Planner decomposes task and creates execution plan\n"
        "2. Researcher retrieves grounded notes with citations\n"
        "3. Writer produces final deliverable using research notes\n"
        "4. Verifier checks for hallucinations, missing evidence, contradictions"
    )
    st.divider()
    st.markdown(
        "**Industry:** Healthcare & Life Sciences\n\n"
        "**Stack:** LangGraph + OpenAI + ChromaDB"
    )
    st.divider()
    st.caption("Prompt injection defense active")

# ─── Pre-fill from sample selection ─────────────────────────────────────────
_prefill_task = ""
_prefill_goal = ""
if st.session_state.selected_sample:
    _prefill_task = st.session_state.selected_sample["task"]
    _prefill_goal = st.session_state.selected_sample["goal"]

# ─── Input form ─────────────────────────────────────────────────────────────
with st.form("task_form"):
    task = st.text_area(
        "Business Request",
        value=_prefill_task,
        placeholder="e.g., Analyze the impact of FDA adaptive trial guidelines on oncology Phase III timelines and recommend improvements.",
        height=100,
    )
    goal = st.text_input(
        "Goal (optional)",
        value=_prefill_goal,
        placeholder="e.g., Produce actionable recommendations for the clinical ops team.",
    )
    submitted = st.form_submit_button("Run Pipeline", use_container_width=True)

# ─── Pipeline execution ────────────────────────────────────────────────────
if submitted and task.strip():
    # Prompt injection defense
    try:
        clean_task, clean_goal = validate_and_sanitize(task.strip(), goal.strip())
    except PromptInjectionError as e:
        st.error(f"Blocked: {e}")
        st.stop()

    mode = output_mode.lower()  # "executive" or "analyst"

    with st.spinner(f"Running multi-agent pipeline ({output_mode} mode) …"):
        start = time.time()
        try:
            result = run_pipeline(task=clean_task, goal=clean_goal, output_mode=mode)
            elapsed = round(time.time() - start, 1)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    st.success(f"Pipeline completed in {elapsed}s")

    # ── Tabs for deliverables ───
    summary_label = "Analyst Summary" if mode == "analyst" else "Executive Summary"
    tab_summary, tab_email, tab_actions, tab_sources, tab_trace, tab_observe = st.tabs(
        [summary_label, "Client Email", "Action Items", "Sources", "Trace Log", "Observability"]
    )

    # Use verified versions if available, else fall back to writer output
    summary = result.get("verified_summary") or result.get("executive_summary", "")
    email = result.get("verified_email") or result.get("client_email", "")
    actions = result.get("verified_action_items") or result.get("action_items", [])
    sources = result.get("verified_sources") or result.get("sources_section", "")

    # ── Summary ───
    summary_heading = "Analyst Summary" if mode == "analyst" else "Executive Summary"
    with tab_summary:
        st.markdown(f"### {summary_heading}")
        st.markdown(summary or "_No summary generated._")
        word_count = len(summary.split()) if summary else 0
        st.caption(f"{word_count} words (max 150)")

    # ── Client Email ───
    with tab_email:
        st.markdown("### Client-Ready Email")
        st.markdown(email or "_No email generated._")

    # ── Action Items ───
    with tab_actions:
        st.markdown("### Action Items")
        if actions:
            df = pd.DataFrame(
                [
                    {
                        "Action": a.action if isinstance(a, ActionItem) else a.get("action", ""),
                        "Owner": a.owner if isinstance(a, ActionItem) else a.get("owner", ""),
                        "Due Date": a.due_date if isinstance(a, ActionItem) else a.get("due_date", ""),
                        "Confidence": a.confidence if isinstance(a, ActionItem) else a.get("confidence", ""),
                    }
                    for a in actions
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No action items generated.")

    # ── Sources ───
    with tab_sources:
        st.markdown("### Sources & Citations")
        st.markdown(sources or "_No sources listed._")

    # ── Observability ───
    with tab_observe:
        st.markdown("### Observability")
        metrics_list: list[AgentMetrics] = result.get("agent_metrics", [])
        if metrics_list:
            rows = []
            for m in metrics_list:
                total_tokens = m.input_tokens + m.output_tokens
                rows.append({
                    "Agent": m.agent.capitalize(),
                    "Latency (s)": m.latency_s,
                    "Input Tokens": m.input_tokens,
                    "Output Tokens": m.output_tokens,
                    "Total Tokens": total_tokens,
                    "Error": m.error or "—",
                })
            df_obs = pd.DataFrame(rows)
            st.dataframe(df_obs, use_container_width=True, hide_index=True)

            # Totals row
            total_latency = sum(m.latency_s for m in metrics_list)
            total_in = sum(m.input_tokens for m in metrics_list)
            total_out = sum(m.output_tokens for m in metrics_list)
            errors = [m for m in metrics_list if m.error]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Latency", f"{total_latency:.1f}s")
            col2.metric("Total Input Tokens", f"{total_in:,}")
            col3.metric("Total Output Tokens", f"{total_out:,}")
            col4.metric("Errors", len(errors))
        else:
            st.info("No observability data available.")

    # ── Trace / Agent log ───
    with tab_trace:
        st.markdown("### Agent Trace Log")
        st.caption(f"Pipeline: Plan → Research → Write → Verify  |  Mode: {mode}")

        # ── 1. Planner ──
        plan = result.get("plan", [])
        with st.expander(f"Planner  —  {len(plan)} sub-questions", expanded=False):
            if plan:
                for i, q in enumerate(plan, 1):
                    st.markdown(f"{i}. {q}")
            else:
                st.info("No execution plan generated.")

        # ── 2. Researcher ──
        notes = result.get("research_notes", [])
        found = [n for n in notes if "not found in sources" not in n.content.lower()]
        gaps = [n for n in notes if "not found in sources" in n.content.lower()]
        with st.expander(
            f"Researcher  —  {len(found)} notes grounded, {len(gaps)} gaps",
            expanded=False,
        ):
            if notes:
                for n in notes:
                    is_gap = "not found in sources" in n.content.lower()
                    icon = "⚠️" if is_gap else "✅"
                    st.markdown(f"{icon} {n.content}")
                    st.caption(f"Citation: {n.citation or 'No citation'}")
            else:
                st.info("No research notes produced.")

        # ── 3. Writer ──
        writer_summary = result.get("executive_summary", "")
        writer_email = result.get("client_email", "")
        writer_actions = result.get("action_items", [])
        writer_produced = bool(writer_summary or writer_email)
        with st.expander(
            f"Writer  —  {'output produced' if writer_produced else 'skipped (no evidence)'}",
            expanded=False,
        ):
            if writer_produced:
                word_count = len(writer_summary.split()) if writer_summary else 0
                st.markdown(f"**Summary:** {word_count} words")
                st.markdown(f"**Email:** {'generated' if writer_email else 'empty'}")
                st.markdown(f"**Action items:** {len(writer_actions)}")
            else:
                st.info("Writer returned empty output (no grounded evidence).")

        # ── 4. Verifier ──
        issues = result.get("verification_issues", [])
        v_passed = result.get("verification_passed")
        verifier_ran = v_passed is not None
        if verifier_ran:
            status = "✅ passed" if v_passed else f"⚠️ {len(issues)} issue(s) found & corrected"
        else:
            status = "skipped"
        with st.expander(f"Verifier  —  {status}", expanded=False):
            if not verifier_ran:
                st.info("Verifier was skipped (writer produced no output).")
            elif issues:
                for issue in issues:
                    text = str(issue)
                    if text.startswith("[Hallucination]"):
                        st.error(f"{text}")
                    elif text.startswith("[Missing Evidence]"):
                        st.warning(f"{text}")
                    elif text.startswith("[Contradiction]"):
                        st.error(f"{text}")
                    else:
                        st.warning(f"{text}")
            else:
                st.success("No issues found — deliverable passed verification.")

elif submitted:
    st.warning("Please enter a business request.")
