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
from datetime import datetime

from agents.graph import run_pipeline
from agents.state import AgentTrace, ActionItem
from agents.guardrails import validate_and_sanitize, PromptInjectionError
from agents.obs_logger import save_run_log, compute_aggregated_metrics, LOGS_DIR

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Multi-Agent Copilot",
    page_icon="+",
    layout="wide",
)

st.title("Enterprise Multi-Agent Copilot")
st.caption("Healthcare & Life Sciences — Plan → Research → Draft → Verify → Deliver")

# ─── Session state ──────────────────────────────────────────────────────────
if "run_history" not in st.session_state:
    st.session_state.run_history = []
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
        "1. Planner decomposes your task\n"
        "2. Researcher retrieves evidence\n"
        "3. Writer drafts the deliverable\n"
        "4. Verifier checks for hallucinations"
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

    # ── Store run in history + persist log ───
    traces: list[AgentTrace] = result.get("traces", [])
    st.session_state.run_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": clean_task[:80],
        "mode": output_mode,
        "elapsed": elapsed,
        "total_tokens_in": sum(t.tokens_in for t in traces),
        "total_tokens_out": sum(t.tokens_out for t in traces),
        "agents": [
            {
                "agent": t.agent,
                "status": t.status,
                "latency_s": t.latency_s,
                "tokens_in": t.tokens_in,
                "tokens_out": t.tokens_out,
                "error": t.error or "",
            }
            for t in traces
        ],
    })

    # Persist per-run observability log to /logs
    run_id = save_run_log(
        task=clean_task,
        goal=clean_goal,
        output_mode=mode,
        elapsed_s=elapsed,
        traces=traces,
        verification_passed=result.get("verification_passed"),
        verification_issues=result.get("verification_issues", []),
    )

    # ── Tabs for deliverables ───
    summary_label = "Analyst Summary" if mode == "analyst" else "Executive Summary"
    tab_summary, tab_email, tab_actions, tab_sources, tab_trace, tab_obs, tab_metrics = st.tabs(
        [summary_label, "Client Email", "Action Items", "Sources", "Trace Log", "Observability", "Aggregated Metrics"]
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

    # ── Trace / Agent log ───
    with tab_trace:
        st.markdown("### Agent Trace Log")

        # Plan
        plan = result.get("plan", [])
        if plan:
            with st.expander("Planner - Execution Plan", expanded=True):
                for i, q in enumerate(plan, 1):
                    st.markdown(f"{i}. {q}")

        # Research notes
        notes = result.get("research_notes", [])
        if notes:
            with st.expander("Researcher - Research Notes", expanded=False):
                for n in notes:
                    st.markdown(f"- {n.content}")
                    st.caption(f"Citation: {n.citation}")

        # Verification issues
        issues = result.get("verification_issues", [])
        if issues:
            with st.expander("Verifier - Issues Found", expanded=True):
                for issue in issues:
                    st.warning(issue)
        else:
            st.info("Verifier found no issues.")

    # ── Observability ───
    with tab_obs:
        st.markdown("### Current Run")
        traces: list[AgentTrace] = result.get("traces", [])
        if traces:
            obs_df = pd.DataFrame(
                [
                    {
                        "Agent": t.agent,
                        "Status": "PASS" if t.status == "success" else "FAIL",
                        "Latency (s)": t.latency_s,
                        "Tokens In": t.tokens_in,
                        "Tokens Out": t.tokens_out,
                        "Error": t.error or "–",
                    }
                    for t in traces
                ]
            )
            st.dataframe(obs_df, use_container_width=True, hide_index=True)

            # Totals
            total_latency = sum(t.latency_s for t in traces)
            total_in = sum(t.tokens_in for t in traces)
            total_out = sum(t.tokens_out for t in traces)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Latency", f"{total_latency:.1f}s")
            col2.metric("Total Tokens In", f"{total_in:,}")
            col3.metric("Total Tokens Out", f"{total_out:,}")
        else:
            st.info("No trace data available.")

        # ── Run History ───
        st.divider()
        st.markdown("### Run History")
        if st.session_state.run_history:
            if st.button("Clear History"):
                st.session_state.run_history = []
                st.rerun()

            for i, run in enumerate(reversed(st.session_state.run_history), 1):
                label = f"Run {len(st.session_state.run_history) - i + 1} — {run['timestamp']} | {run['mode']} | {run['elapsed']}s | {run['task'][:50]}…"
                with st.expander(label, expanded=(i == 1)):
                    run_df = pd.DataFrame(
                        [
                            {
                                "Agent": a["agent"],
                                "Status": "PASS" if a["status"] == "success" else "FAIL",
                                "Latency (s)": a["latency_s"],
                                "Tokens In": a["tokens_in"],
                                "Tokens Out": a["tokens_out"],
                                "Error": a["error"] or "–",
                            }
                            for a in run["agents"]
                        ]
                    )
                    st.dataframe(run_df, use_container_width=True, hide_index=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Latency", f"{run['elapsed']}s")
                    c2.metric("Tokens In", f"{run['total_tokens_in']:,}")
                    c3.metric("Tokens Out", f"{run['total_tokens_out']:,}")
        else:
            st.info("No previous runs yet.")

    # ── Aggregated Metrics ───
    with tab_metrics:
        st.markdown("### Aggregated System Metrics")
        st.caption(f"Computed from all runs saved in `/logs` ({len(list(LOGS_DIR.glob('*.json')))} log files)")

        # Clear logs button
        col1, col2 = st.columns([3, 1])
        with col1:
            pass  # Empty for spacing
        with col2:
            if st.button("🗑️ Clear Logs", help="Delete all observability log files"):
                import shutil
                if LOGS_DIR.exists():
                    shutil.rmtree(LOGS_DIR)
                    LOGS_DIR.mkdir(exist_ok=True)
                st.success("All logs cleared!")
                st.rerun()

        metrics = compute_aggregated_metrics()

        if metrics["total_runs"] == 0:
            st.info("No historical logs found. Run the pipeline to start collecting metrics.")
        else:
            # ── Top-level KPIs ───
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Runs", metrics["total_runs"])
            k2.metric("Pipeline Success", f"{metrics['success_rate_pct']}%")
            k3.metric("Avg Latency", f"{metrics['avg_elapsed_s']}s")
            verified_corrected = metrics['verification_failed']
            k4.metric("Verifier Corrections", f"{verified_corrected} / {metrics['total_runs']}")

            st.divider()

            # ── Latency breakdown ───
            st.markdown("#### Latency")
            l1, l2, l3 = st.columns(3)
            l1.metric("Average", f"{metrics['avg_elapsed_s']}s")
            l2.metric("Minimum", f"{metrics['min_elapsed_s']}s")
            l3.metric("Maximum", f"{metrics['max_elapsed_s']}s")

            # ── Token usage ───
            st.markdown("#### Token Usage")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Total Tokens In", f"{metrics['total_tokens_in']:,}")
            t2.metric("Total Tokens Out", f"{metrics['total_tokens_out']:,}")
            t3.metric("Avg In / Run", f"{metrics['avg_tokens_in_per_run']:,}")
            t4.metric("Avg Out / Run", f"{metrics['avg_tokens_out_per_run']:,}")

            st.divider()

            # ── Per-Agent Performance ───
            st.markdown("#### Per-Agent Performance (averages across all runs)")
            agent_data = metrics.get("per_agent", {})
            if agent_data:
                agent_order = ["Planner", "Researcher", "Writer", "Verifier"]
                sorted_agents = sorted(agent_data.items(), key=lambda x: agent_order.index(x[0]) if x[0] in agent_order else 99)
                agent_rows = []
                for name, v in sorted_agents:
                    agent_rows.append({
                        "Agent": name,
                        "Runs": v["runs"],
                        "Avg Latency (s)": v["avg_latency_s"],
                        "Avg Tokens In": v["avg_tokens_in"],
                        "Avg Tokens Out": v["avg_tokens_out"],
                        "Errors": v["error_count"],
                        "Error Rate": f"{v['error_rate_pct']}%",
                    })
                st.dataframe(pd.DataFrame(agent_rows), use_container_width=True, hide_index=True)

            st.divider()

            # ── Mode distribution ───
            st.markdown("#### Output Mode Distribution")
            mode_data = metrics.get("mode_distribution", {})
            if mode_data:
                mode_df = pd.DataFrame(
                    [{"Mode": m.title(), "Runs": c, "Share": f"{c / metrics['total_runs'] * 100:.0f}%"} for m, c in mode_data.items()]
                )
                st.dataframe(mode_df, use_container_width=True, hide_index=True)

            # ── Verification stats ───
            st.markdown("#### Verifier Activity")
            v1, v2, v3 = st.columns(3)
            v1.metric("Clean Drafts", metrics["verification_passed"])
            v2.metric("Corrected Drafts", metrics["verification_failed"])
            clean_pct = metrics['verification_rate_pct']
            v3.metric("Clean Draft Rate", f"{clean_pct}%")
            st.caption("Clean Draft = Writer output passed verification with no issues. "
                       "Corrected Draft = Verifier found and fixed unsupported claims (normal operation).")

elif submitted:
    st.warning("Please enter a business request.")
