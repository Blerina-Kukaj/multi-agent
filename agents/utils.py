"""
Shared utilities for agent modules – JSON parsing, code fence removal,
and evidence checking.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.state import ResearchNote


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_llm_json(raw: str) -> Any:
    """Parse JSON from LLM output, handling common issues.

    - Strips code fences
    - Fixes trailing commas
    - Uses json.JSONDecoder.raw_decode for robust extraction
    """
    cleaned = strip_code_fences(raw.strip())

    # Fix trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Try direct parse first (fastest path)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to raw_decode to find first valid JSON object/array
    decoder = json.JSONDecoder()
    for marker in ("{", "["):
        if marker in cleaned:
            try:
                idx = cleaned.index(marker)
                obj, _ = decoder.raw_decode(cleaned, idx)
                return obj
            except (json.JSONDecodeError, ValueError):
                continue

    raise json.JSONDecodeError("No valid JSON found in LLM output", cleaned, 0)


def has_real_evidence(notes: list[ResearchNote]) -> bool:
    """Return True if at least one research note has real sourced content."""
    if not notes:
        return False
    for n in notes:
        text = n.content.lower()
        if "not found in sources" not in text and "additional information needed" not in text:
            return True
    return False
