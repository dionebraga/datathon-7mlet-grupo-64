"""Pluggable LLM client with an offline deterministic fallback.

Providers:
* ``offline``  — a deterministic, template-based summariser. No network, no keys.
  It only restates grounded facts (decision fields + retrieved snippets), so it
  never hallucinates. This is the default and is what CI uses.
* ``anthropic``— Claude via the official SDK (if installed and ``ANTHROPIC_API_KEY``
  is set). Used for higher-quality natural-language explanations.

The target Azure architecture maps this to Azure OpenAI behind the same
interface (see docs/architecture-azure.md).
"""

from __future__ import annotations

import os
import textwrap

from adaptive_offers.config import get_settings
from adaptive_offers.logging_utils import get_logger

logger = get_logger("assistant.llm")


class LLMClient:
    def __init__(self, provider: str | None = None) -> None:
        settings = get_settings()
        self.provider = (provider or settings.llm_provider or "offline").lower()
        self.model = settings.anthropic_model
        if self.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.info('{"event": "llm_fallback", "reason": "missing ANTHROPIC_API_KEY"}')
            self.provider = "offline"

    def complete(self, prompt: str, system: str = "", grounding: str = "") -> str:
        if self.provider == "anthropic":
            try:
                return self._anthropic(prompt, system)
            except Exception as exc:  # pragma: no cover - network/optional dep
                logger.info('{"event": "llm_error", "provider": "anthropic", "err": "%s"}', exc)
                return self._offline(prompt, grounding)
        return self._offline(prompt, grounding)

    # --- providers ----------------------------------------------------------
    def _anthropic(self, prompt: str, system: str) -> str:  # pragma: no cover - optional
        import anthropic  # type: ignore

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=600,
            system=system or "Você é um assistente de governança de ML. Responda apenas com base no contexto fornecido.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")

    def _offline(self, prompt: str, grounding: str) -> str:
        """Deterministic, grounded summary — restates facts, never invents."""
        body = grounding.strip() or prompt.strip()
        header = "Resumo (modo offline determinístico — sem LLM externo):"
        wrapped = textwrap.fill(body, width=100, replace_whitespace=False)
        return f"{header}\n{wrapped}"
