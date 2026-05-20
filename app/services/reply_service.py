"""Compatibility shim for legacy imports."""

from app.services.reply.reply_service import *  # noqa: F401,F403

__all__ = ["ReplyGenerator"]
