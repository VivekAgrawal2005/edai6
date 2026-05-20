"""Test configuration for the email intelligence backend."""

from __future__ import annotations

import os


os.environ.setdefault("DATABASE_URL", "sqlite:///./test_email_intelligence.db")
os.environ.setdefault("USE_HF_SPAM_CLASSIFIER", "false")
os.environ.setdefault("HF_SPAM_DEVICE", "-1")
