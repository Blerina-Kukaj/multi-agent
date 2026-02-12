"""
Prompt Injection Defense – validates and sanitizes user input
before it enters the multi-agent pipeline.

Implements pattern-based detection and input sanitization rules.
"""

from __future__ import annotations

import re

# ── Known injection patterns ───────────────────────────────────────────────
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Direct override attempts
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
     "Attempted to override system instructions"),
    (r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
     "Attempted to disregard system instructions"),
    (r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
     "Attempted to override system instructions"),
    # Role hijacking
    (r"you\s+are\s+now\s+(a|an|the)\s+",
     "Attempted role hijacking"),
    (r"act\s+as\s+(a|an|the)\s+",
     "Attempted role hijacking"),
    (r"pretend\s+(you\s+are|to\s+be)\s+",
     "Attempted role impersonation"),
    # System prompt extraction
    (r"(show|reveal|print|display|output|repeat)\s+.{0,10}(your|the|system)\s+(prompt|instructions?|rules?)",
     "Attempted system prompt extraction"),
    (r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
     "Attempted system prompt extraction"),
    # Delimiter injection
    (r"```\s*(system|assistant|user)\s*\n",
     "Attempted delimiter injection"),
    (r"<\s*/?\s*(system|prompt|instruction)",
     "Attempted XML/tag injection"),
    # Encoding evasion
    (r"base64\s*(encode|decode)",
     "Attempted encoding evasion"),
    # Data exfiltration
    (r"(send|post|fetch|curl|wget|http)\s+.*(api|endpoint|url|webhook)",
     "Attempted data exfiltration"),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in _INJECTION_PATTERNS
]

# ── Maximum input length ───────────────────────────────────────────────────
MAX_INPUT_LENGTH = 2000  # characters


class PromptInjectionError(Exception):
    """Raised when a prompt injection attempt is detected."""
    pass


def check_injection(text: str) -> tuple[bool, str]:
    """
    Check text for prompt injection patterns.

    Returns
    -------
    (is_safe, reason) : tuple[bool, str]
        is_safe=True if no injection detected, else False with reason.
    """
    if not text or not text.strip():
        return True, ""

    for pattern, reason in _COMPILED_PATTERNS:
        if pattern.search(text):
            return False, reason

    return True, ""


def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing potentially dangerous content.

    - Strips excessive whitespace
    - Removes control characters
    - Truncates to MAX_INPUT_LENGTH
    """
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Truncate
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH] + "..."
    return text.strip()


def validate_and_sanitize(task: str, goal: str = "") -> tuple[str, str]:
    """
    Full validation pipeline for user inputs.

    Returns
    -------
    (sanitized_task, sanitized_goal) : tuple[str, str]

    Raises
    ------
    PromptInjectionError
        If injection is detected in either field.
    """
    # Check for injections
    safe, reason = check_injection(task)
    if not safe:
        raise PromptInjectionError(f"Input rejected: {reason}")

    safe, reason = check_injection(goal)
    if not safe:
        raise PromptInjectionError(f"Goal rejected: {reason}")

    # Sanitize
    return sanitize_input(task), sanitize_input(goal)
