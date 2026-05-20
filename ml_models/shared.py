"""Shared helpers for training email intelligence classifiers from Enron data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from app.config import settings
from app.services.heuristics import (
    compose_text,
    weak_label_intent,
    weak_label_reply_needed,
    weak_label_spam,
)
import numpy as np
import re


MESSAGE_HEADERS = re.compile(
    r"^(From|To|Cc|Bcc|Subject|Date|Sent|Received|Message-ID|X-[^:]+|Mime-Version|Content-Type|Content-Transfer-Encoding):\s*(.*)$",
    re.IGNORECASE,
)


def parse_message(message: str) -> dict:
    sender = None
    subject = None
    body_lines = []
    in_body = False
    for line in message.splitlines():
        if not in_body:
            match = MESSAGE_HEADERS.match(line)
            if match:
                key = match.group(1).lower()
                value = match.group(2).strip()
                if key == "from":
                    sender = value
                elif key == "subject":
                    subject = value
                elif key == "message-id":
                    continue
                else:
                    continue
            if line.strip() == "":
                in_body = True
                continue
            if subject is None and line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    if not body:
        body = message
    return {"sender": sender, "subject": subject, "body": body}


def load_dataset(max_rows: int | None = None) -> pd.DataFrame:
    path = Path(settings.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, usecols=["message"], nrows=max_rows, on_bad_lines="skip")
    parsed = frame["message"].dropna().astype(str).apply(parse_message)
    records = pd.DataFrame(parsed.tolist())
    records["text"] = records.apply(lambda row: compose_text(row["subject"], row["body"]), axis=1)
    records["spam_label"] = records.apply(lambda row: weak_label_spam(row["subject"], row["body"]), axis=1)
    records["reply_label"] = records.apply(
        lambda row: weak_label_reply_needed(row["sender"], row["subject"], row["body"]), axis=1
    )
    records["intent_label"] = records.apply(lambda row: weak_label_intent(row["subject"], row["body"]), axis=1)
    return records.dropna(subset=["text"]).reset_index(drop=True)


def engineered_features_from_texts(texts: Iterable[str]) -> np.ndarray:
    """Compute simple engineered numeric features from raw text.

    Features:
    - num_urls
    - num_currency_symbols
    - num_digits
    - exclamation_count
    - uppercase_ratio
    - has_promo_keyword (0/1)
    """
    urls_re = re.compile(r"https?://|www\.", re.I)
    promo_re = re.compile(r"\b(discount|offer|% off|limited time|buy now|free|unsubscribe|sale)\b", re.I)
    features = []
    for text in texts:
        t = str(text or "")
        num_urls = len(urls_re.findall(t))
        num_currency = len(re.findall(r"[$€£]", t))
        num_digits = sum(c.isdigit() for c in t)
        exclaim = t.count("!")
        up_ratio = sum(1 for c in t if c.isupper()) / max(1, len(t))
        has_promo = 1 if promo_re.search(t) else 0
        features.append([num_urls, num_currency, num_digits, exclaim, up_ratio, has_promo])
    return np.asarray(features, dtype=float)
