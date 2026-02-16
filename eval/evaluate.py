"""
Evaluation runner – executes each test prompt through the pipeline and
reports pass/fail against acceptance criteria.

Usage:  python -m eval.evaluate
"""

from __future__ import annotations

import sys
import os
import json
import time
import re

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.test_prompts import TEST_PROMPTS
from agents.graph import run_pipeline
from agents.state import ActionItem


# ── Evaluation checks ──────────────────────────────────────────────────────
def check_has_summary(result: dict) -> bool:
    summary = result.get("verified_summary") or result.get("executive_summary", "")
    return bool(summary.strip())


def check_summary_length(result: dict) -> bool:
    summary = result.get("verified_summary") or result.get("executive_summary", "")
    return len(summary.split()) <= 160  # 150 + small tolerance


def check_summary_min_length(result: dict) -> bool:
    """Summary should have substance – at least 40 words."""
    summary = result.get("verified_summary") or result.get("executive_summary", "")
    return len(summary.split()) >= 40


def check_has_email(result: dict) -> bool:
    email = result.get("verified_email") or result.get("client_email", "")
    return bool(email.strip())


def check_email_professional(result: dict) -> bool:
    """Email must have a greeting and sign-off to be client-ready."""
    email = (result.get("verified_email") or result.get("client_email", "")).lower()
    if not email.strip():
        return False
    has_greeting = any(g in email for g in ["dear", "hi ", "hello", "good morning", "good afternoon"])
    has_signoff = any(s in email for s in ["regards", "sincerely", "best", "thank you", "thanks"])
    return has_greeting and has_signoff


def check_has_action_items(result: dict) -> bool:
    items = result.get("verified_action_items") or result.get("action_items", [])
    return len(items) > 0


def check_action_items_complete(result: dict) -> bool:
    """Every action item must have owner, due_date, confidence populated (not blank/TBD)."""
    items = result.get("verified_action_items") or result.get("action_items", [])
    if not items:
        return False
    for a in items:
        owner = a.owner if isinstance(a, ActionItem) else a.get("owner", "")
        due = a.due_date if isinstance(a, ActionItem) else a.get("due_date", "")
        conf = a.confidence if isinstance(a, ActionItem) else a.get("confidence", "")
        # All three fields must be present and not just "TBD" or empty
        if not owner or owner.strip().upper() == "TBD":
            return False
        if not due or due.strip().upper() == "TBD":
            return False
        if not conf or conf.strip().upper() == "TBD":
            return False
    return True


def _normalize_sources(sources) -> str:
    """Ensure sources is always a string, regardless of type."""
    if isinstance(sources, str):
        return sources
    if isinstance(sources, list):
        return "\n".join(str(s) for s in sources)
    if isinstance(sources, dict):
        return json.dumps(sources)
    return str(sources)


def check_has_citations(result: dict) -> bool:
    sources = result.get("verified_sources") or result.get("sources_section", "")
    sources = _normalize_sources(sources)
    # Check for citation pattern [filename | Chunk #N]
    return bool(re.search(r"\[.+?\|.+?\]", sources))



def check_verifier_ran(result: dict) -> bool:
    return "verification_passed" in result


def check_no_empty_sections(result: dict) -> bool:
    summary = result.get("verified_summary") or result.get("executive_summary", "")
    email = result.get("verified_email") or result.get("client_email", "")
    sources = result.get("verified_sources") or result.get("sources_section", "")
    sources = _normalize_sources(sources)
    return all(s.strip() for s in [summary, email, sources])


CHECKS = [
    ("Has executive summary", check_has_summary),
    ("Summary ≤ 150 words", check_summary_length),
    ("Summary ≥ 40 words", check_summary_min_length),
    ("Has client email", check_has_email),
    ("Email is professional", check_email_professional),
    ("Has action items", check_has_action_items),
    ("Action items complete", check_action_items_complete),
    ("Has citations", check_has_citations),
    ("Verifier ran", check_verifier_ran),
    ("No empty sections", check_no_empty_sections),
]


def evaluate_one(prompt: dict, index: int) -> dict:
    """Run one test prompt and evaluate it."""
    print(f"\n{'='*70}")
    print(f"Test {index + 1}: {prompt['task'][:80]}…")
    print(f"{'='*70}")

    start = time.time()
    try:
        result = run_pipeline(task=prompt["task"], goal=prompt["goal"])
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"  PIPELINE ERROR ({elapsed}s): {e}")
        return {"index": index + 1, "status": "error", "error": str(e), "elapsed": elapsed}

    elapsed = round(time.time() - start, 1)

    results = {}
    all_passed = True
    for name, check_fn in CHECKS:
        passed = check_fn(result)
        results[name] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print(f"  Time: {elapsed}s")
    return {
        "index": index + 1,
        "task": prompt["task"][:60],
        "status": "pass" if all_passed else "fail",
        "checks": results,
        "elapsed": elapsed,
    }


def main() -> None:
    print("Enterprise Multi-Agent Copilot - Evaluation Suite")
    print(f"   Running {len(TEST_PROMPTS)} test prompts\n")

    all_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        result = evaluate_one(prompt, i)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in all_results if r["status"] == "pass")
    failed = sum(1 for r in all_results if r["status"] == "fail")
    errors = sum(1 for r in all_results if r["status"] == "error")
    total_time = sum(r.get("elapsed", 0) for r in all_results)

    print(f"  Passed:  {passed}/{len(all_results)}")
    print(f"  Failed:  {failed}/{len(all_results)}")
    print(f"  Errors:  {errors}/{len(all_results)}")
    print(f"  Total:   {total_time:.1f}s")

    # Write results to file
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
