"""
Observability Logger – persists per-run JSON logs and computes aggregated
metrics across all historical runs for enterprise traceability.

Logs are written to /logs as individual JSON files (one per pipeline run).
The aggregator reads all log files to produce system-level KPIs.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.state import AgentTrace

# ── Log directory ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = _PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ── Per-Run Log ──────────────────────────────────────────────────────────────

def save_run_log(
    *,
    task: str,
    goal: str,
    output_mode: str,
    elapsed_s: float,
    traces: list[AgentTrace],
    verification_passed: bool | None = None,
    verification_issues: list[str] | None = None,
) -> str:
    """
    Persist a single pipeline run as a JSON file for enterprise traceability.

    Returns the generated run_id.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    timestamp = datetime.now().isoformat()

    agent_records = []
    for t in traces:
        agent_records.append({
            "agent": t.agent,
            "status": t.status,
            "latency_s": t.latency_s,
            "tokens_in": t.tokens_in,
            "tokens_out": t.tokens_out,
            "error": t.error or None,
        })

    log_entry: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "task": task,
        "goal": goal,
        "output_mode": output_mode,
        "elapsed_s": elapsed_s,
        "agents": agent_records,
        "total_tokens_in": sum(t.tokens_in for t in traces),
        "total_tokens_out": sum(t.tokens_out for t in traces),
        "total_latency_s": round(sum(t.latency_s for t in traces), 2),
        "verification_passed": verification_passed,
        "verification_issues": verification_issues or [],
        "agent_count": len(traces),
        "all_agents_passed": all(t.status == "success" for t in traces),
    }

    log_path = LOGS_DIR / f"{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    return run_id


# ── Load all logs ────────────────────────────────────────────────────────────

def load_all_logs() -> list[dict[str, Any]]:
    """Read every JSON log file from /logs, sorted by timestamp ascending."""
    logs: list[dict[str, Any]] = []
    if not LOGS_DIR.exists():
        return logs
    for fpath in sorted(LOGS_DIR.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                logs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue  # skip corrupted files
    return logs


# ── Aggregated Metrics ───────────────────────────────────────────────────────

def compute_aggregated_metrics() -> dict[str, Any]:
    """
    Compute system-level performance metrics across all historical runs.

    Returns a dictionary with KPIs that enable system-level evaluation:
      - total_runs, success_rate, avg/min/max latency, token usage,
        per-agent averages, mode distribution, verification stats.
    """
    logs = load_all_logs()
    if not logs:
        return {"total_runs": 0}

    total = len(logs)
    successful = sum(1 for lg in logs if lg.get("all_agents_passed", False))
    verified = sum(1 for lg in logs if lg.get("verification_passed") is True)
    verify_failed = sum(1 for lg in logs if lg.get("verification_passed") is False)

    elapsed_vals = [lg["elapsed_s"] for lg in logs]
    tokens_in_vals = [lg["total_tokens_in"] for lg in logs]
    tokens_out_vals = [lg["total_tokens_out"] for lg in logs]

    # Per-agent breakdown
    agent_stats: dict[str, dict[str, list]] = {}
    for lg in logs:
        for a in lg.get("agents", []):
            name = a["agent"]
            if name not in agent_stats:
                agent_stats[name] = {"latency": [], "tokens_in": [], "tokens_out": [], "errors": 0}
            agent_stats[name]["latency"].append(a["latency_s"])
            agent_stats[name]["tokens_in"].append(a["tokens_in"])
            agent_stats[name]["tokens_out"].append(a["tokens_out"])
            if a.get("error"):
                agent_stats[name]["errors"] += 1

    per_agent = {}
    for name, stats in agent_stats.items():
        n = len(stats["latency"])
        per_agent[name] = {
            "runs": n,
            "avg_latency_s": round(sum(stats["latency"]) / n, 2) if n else 0,
            "avg_tokens_in": round(sum(stats["tokens_in"]) / n) if n else 0,
            "avg_tokens_out": round(sum(stats["tokens_out"]) / n) if n else 0,
            "error_count": stats["errors"],
            "error_rate_pct": round(stats["errors"] / n * 100, 1) if n else 0,
        }

    # Mode distribution
    mode_counts: dict[str, int] = {}
    for lg in logs:
        m = lg.get("output_mode", "unknown")
        mode_counts[m] = mode_counts.get(m, 0) + 1

    # Hourly distribution (for usage pattern insight)
    hour_counts: dict[int, int] = {}
    for lg in logs:
        try:
            h = datetime.fromisoformat(lg["timestamp"]).hour
            hour_counts[h] = hour_counts.get(h, 0) + 1
        except (KeyError, ValueError):
            pass

    return {
        "total_runs": total,
        "successful_runs": successful,
        "success_rate_pct": round(successful / total * 100, 1),
        "verification_passed": verified,
        "verification_failed": verify_failed,
        "verification_rate_pct": round(verified / total * 100, 1) if total else 0,
        "avg_elapsed_s": round(sum(elapsed_vals) / total, 2),
        "min_elapsed_s": round(min(elapsed_vals), 2),
        "max_elapsed_s": round(max(elapsed_vals), 2),
        "total_tokens_in": sum(tokens_in_vals),
        "total_tokens_out": sum(tokens_out_vals),
        "avg_tokens_in_per_run": round(sum(tokens_in_vals) / total),
        "avg_tokens_out_per_run": round(sum(tokens_out_vals) / total),
        "per_agent": per_agent,
        "mode_distribution": mode_counts,
        "hourly_distribution": hour_counts,
    }
