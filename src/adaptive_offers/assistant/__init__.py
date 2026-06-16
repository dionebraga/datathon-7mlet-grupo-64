"""LLM + RAG decision assistant.

Implements the Datathon's required assistant: it summarises experiments,
retrieves synthetic internal policies (RAG) and explains decisions. The LLM is
pluggable (Anthropic Claude / Azure OpenAI) and falls back to a deterministic
offline summariser so it runs with zero API keys.
"""

from __future__ import annotations

from adaptive_offers.assistant.explain import (
    Assistant,
    explain_decision,
    summarize_experiment,
)
from adaptive_offers.assistant.llm import LLMClient
from adaptive_offers.assistant.rag import PolicyRAG

__all__ = [
    "Assistant",
    "explain_decision",
    "summarize_experiment",
    "LLMClient",
    "PolicyRAG",
]
