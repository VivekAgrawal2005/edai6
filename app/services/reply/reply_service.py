"""Deterministic template-based reply generation.

Canonical location for reply generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.services.heuristics import detect_time_context, extract_sender_name
from app.services.ollama_service import OllamaClient
from app.services.retriever import Retriever


class ReplyGenerator:
    def __init__(self, templates_path: str | None = None) -> None:
        self.templates_path = Path(templates_path or settings.templates_path)
        self.templates = self._load_templates()
        self._use_ollama = settings.use_ollama
        self._ollama = OllamaClient() if self._use_ollama else None
        self._retriever = Retriever()

    def _load_templates(self) -> Dict[str, List[str]]:
        if not self.templates_path.exists():
            return {}
        with self.templates_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _stable_choice_index(self, intent: str, seed: str) -> int:
        templates = self.templates.get(intent) or self.templates.get("fallback", [])
        if not templates:
            return 0
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return int(digest, 16) % len(templates)

    def _context_values(self, sender: str | None, subject: str | None, body: str | None) -> Dict[str, str]:
        text = f"{subject or ''} {body or ''}"
        return {
            "sender_name": extract_sender_name(sender),
            "time": detect_time_context(text) or "the proposed time",
            "subject": (subject or "").strip(),
            "date": detect_time_context(text) or "soon",
        }

    def generate(
        self,
        intent: str,
        sender: str | None = None,
        subject: str | None = None,
        body: str | None = None,
        email_id: str | None = None,
    ) -> Optional[str]:
        # blocked intents handled by rule engine earlier; keep conservative checks
        if not intent or intent in {"newsletter", "spam"}:
            return None

        # If Ollama enabled, prefer LLM-generated reply using RAG examples
        if self._use_ollama and self._ollama is not None:
            try:
                examples = self._retriever.retrieve(subject, body)
                # use sync call to avoid event-loop issues in route handlers
                reply = self._ollama.generate_reply_sync(subject, body, intent, examples=examples)
                if reply:
                    return reply
            except Exception:
                # fall back to template approach on any LLM failure
                pass

        # deterministic template fallback
        options = self.templates.get(intent) or self.templates.get("fallback", [])
        if not options:
            return None
        seed = email_id or f"{intent}:{sender}:{subject}:{body}"
        template = options[self._stable_choice_index(intent, seed)]
        context = self._context_values(sender, subject, body)
        try:
            return template.format(**context)
        except KeyError:
            return template


__all__ = ["ReplyGenerator"]
