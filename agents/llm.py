"""
Shared LLM factory – returns cached ChatOpenAI instances to avoid
re-creating HTTP connections and authentication on every agent call.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

import config


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.2, max_tokens: int | None = None) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance keyed by temperature + max_tokens."""
    kwargs: dict = {
        "model": config.OPENAI_MODEL,
        "api_key": config.OPENAI_API_KEY,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)
