"""Business rules for spam blocking and reply suppression.

Canonical location for rule evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


BLOCKED_INTENTS = {"newsletter", "invoice"}


@dataclass
class RuleDecision:
    allow_reply: bool
    reasons: List[str] = field(default_factory=list)


class RuleEngine:
    def should_generate_reply(self, spam: bool, reply_needed: bool, intent: Optional[str]) -> RuleDecision:
        reasons: List[str] = []
        if spam:
            reasons.append("spam_detected")
        if not reply_needed:
            reasons.append("reply_not_needed")
        if intent in BLOCKED_INTENTS:
            reasons.append(f"blocked_intent:{intent}")
        return RuleDecision(not reasons, reasons)

    def apply(self, spam: bool, reply_needed: bool, intent: Optional[str], generated_reply: Optional[str]) -> Optional[str]:
        decision = self.should_generate_reply(spam, reply_needed, intent)
        return generated_reply if decision.allow_reply else None


__all__ = ["RuleDecision", "RuleEngine"]
