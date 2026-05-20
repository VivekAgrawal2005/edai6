"""Compatibility wrapper for legacy imports."""

from app.services.rules.rules_service import *  # noqa: F401,F403

__all__ = ["RuleDecision", "RuleEngine"]

