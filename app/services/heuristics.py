"""Deterministic NLP heuristics used for weak labels and runtime fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple


SPAM_KEYWORDS = {
    "free",
    "winner",
    "urgent",
    "click",
    "prize",
    "credit",
    "loan",
    "viagra",
    "bitcoin",
    "unsubscribe",
    "limited time",
    "act now",
    "buy now",
    "risk free",
}

SPAM_KEYWORD_WEIGHTS = {
    "free": 0.10,
    "winner": 0.12,
    "won": 0.12,
    "congratulations": 0.10,
    "claim now": 0.14,
    "click": 0.08,
    "click here": 0.12,
    "urgent": 0.10,
    "limited offer": 0.14,
    "reward": 0.10,
    "voucher": 0.10,
    "buy now": 0.12,
    "crypto": 0.10,
    "investment opportunity": 0.12,
    "guaranteed": 0.10,
    "earn money": 0.12,
    "discount": 0.08,
    "offer expires": 0.12,
    "gift": 0.08,
}

PHISHING_PATTERNS = (
    r"verify your account",
    r"suspend(ed)? account",
    r"click the link",
    r"confirm your password",
    r"secure your account",
    r"update payment",
    r"login to continue",
)
PROMO_KEYWORDS = {
    "discount",
    "offer",
    "% off",
    "sale",
    "coupon",
    "deal",
    "limited offer",
}

REPLY_KEYWORDS = {
    "please",
    "can you",
    "could you",
    "kindly",
    "let me know",
    "confirm",
    "schedule",
    "meeting",
    "support",
    "help",
    "question",
    "need",
    "follow up",
    "response",
}

INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "meeting_request": ("meeting", "call", "discussion", "sync up", "catch up"),
    "support_request": ("support", "issue", "bug", "error", "problem", "help desk"),
    "complaint": ("complaint", "frustrated", "disappointed", "terrible", "bad service"),
    "follow_up": ("following up", "follow up", "checking in", "reminder", "status"),
    "greeting": ("hello", "hi ", "good morning", "good afternoon", "good evening"),
    "scheduling": ("schedule", "calendar", "available", "availability", "reschedule"),
    "information_request": ("information", "details", "share", "send me", "provide"),
    "thank_you": ("thank you", "thanks", "appreciate"),
    "invoice": ("invoice", "payment", "billing", "receipt", "amount due"),
    "newsletter": ("newsletter", "unsubscribe", "promotion", "marketing", "digest"),
    "job_inquiry": ("job", "career", "position", "application", "resume", "cv"),
}

DEFAULT_INTENT = "follow_up"


def compose_text(subject: str | None, body: str | None) -> str:
    parts = [subject or "", body or ""]
    return " \n ".join(part for part in parts if part).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_sender_name(sender: str | None) -> str:
    if not sender:
        return "there"
    if "@" in sender:
        sender = sender.split("@", 1)[0]
    sender = sender.replace(".", " ").replace("_", " ").strip()
    if not sender:
        return "there"
    return " ".join(part.capitalize() for part in sender.split())


def detect_time_context(text: str) -> str | None:
    lowered = text.lower()
    patterns = [
        r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(tomorrow|today|next week|this week|morning|afternoon|evening)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(0)
    return None


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    lowered = normalize_text(text)
    return sum(1 for keyword in keywords if keyword in lowered)


def spam_heuristic_score(subject: str | None, body: str | None) -> float:
    text = compose_text(subject, body)
    lowered = normalize_text(text)
    if not lowered:
        return 0.05
    score = 0.05
    keyword_hits = count_keyword_hits(lowered, SPAM_KEYWORDS)
    score += min(0.55, keyword_hits * 0.15)
    # promotional keywords increase spaminess
    promo_hits = count_keyword_hits(lowered, PROMO_KEYWORDS)
    score += min(0.30, promo_hits * 0.18)
    score += min(0.15, lowered.count("http") * 0.05)
    score += min(0.10, lowered.count("$") * 0.03)
    score += min(0.10, lowered.count("!") * 0.02)
    if keyword_hits:
        score += 0.10
    if "unsubscribe" in lowered:
        score += 0.15
    # percentages and explicit percent-off patterns
    if re.search(r"\b\d{1,3}%\b", lowered):
        score += 0.12
    return max(0.01, min(score, 0.99))


def spam_rule_score(subject: str | None, body: str | None) -> tuple[float, list[str]]:
    text = compose_text(subject, body)
    lowered = normalize_text(text)
    score = 0.0
    reasons: list[str] = []
    if not lowered:
        return 0.0, reasons

    keyword_hits = []
    for keyword, weight in SPAM_KEYWORD_WEIGHTS.items():
        if keyword in lowered:
            score += weight
            keyword_hits.append(keyword)
    if keyword_hits:
        reasons.append("Contains spam keywords")

    if re.search(r"\b\d{1,3}%\b", lowered):
        score += 0.08
        reasons.append("Contains promotional percent-off language")

    if "unsubscribe" in lowered:
        score += 0.12
        reasons.append("Contains unsubscribe language")

    if re.search(r"http://|https://|www\.", lowered):
        score += 0.10
        reasons.append("Contains suspicious URL")

    if any(pattern in lowered for pattern in PHISHING_PATTERNS):
        score += 0.14
        reasons.append("Contains phishing-style language")

    return max(0.0, min(score, 1.0)), reasons


def spam_heuristic_signals(sender: str | None, subject: str | None, body: str | None) -> tuple[float, list[str]]:
    text = compose_text(subject, body)
    lowered = normalize_text(text)
    score = 0.0
    reasons: list[str] = []
    if not lowered:
        return 0.0, reasons

    subject_text = normalize_text(subject or "")
    body_text = normalize_text(body or "")

    if subject and subject.isupper() and len(subject) >= 6:
        score += 0.10
        reasons.append("ALL CAPS subject")

    if text.count("!") > 3:
        score += 0.10
        reasons.append("Too many exclamation marks")

    if re.search(r"http://|https://|www\.", lowered):
        score += 0.10
        if "Suspicious URL" not in reasons:
            reasons.append("Suspicious URL")

    if re.search(r"\b(urgent|immediately|act now|limited time|claim now|verify now)\b", lowered):
        score += 0.12
        reasons.append("Excessive urgency language")

    words = re.findall(r"\b\w+\b", lowered)
    repeated = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    if repeated >= 2:
        score += 0.08
        reasons.append("Repeated words detected")

    if sender:
        sender_lower = sender.lower()
        suspicious_domains = ("cheap", "promo", "offer", "deal", "win", "reward", "click")
        if any(marker in sender_lower for marker in suspicious_domains):
            score += 0.12
            reasons.append("Suspicious sender domain")

    if any(marker in body_text for marker in ("password", "login", "verify", "bank", "account")):
        score += 0.08
        reasons.append("Phishing-style language")

    if any(marker in subject_text for marker in ("free", "win", "gift", "discount", "offer")):
        score += 0.08

    return max(0.0, min(score, 1.0)), reasons


def reply_needed_heuristic_score(sender: str | None, subject: str | None, body: str | None) -> float:
    text = compose_text(subject, body)
    lowered = normalize_text(text)
    if not lowered:
        return 0.10
    score = 0.10
    score += min(0.30, count_keyword_hits(lowered, REPLY_KEYWORDS) * 0.07)
    if "?" in text:
        score += 0.20
    if any(word in lowered for word in ("schedule", "meeting", "confirm", "please respond")):
        score += 0.20
    if any(word in lowered for word in ("newsletter", "receipt", "invoice")):
        score -= 0.30
    if sender and "noreply" in sender.lower():
        score -= 0.20
    return max(0.01, min(score, 0.99))


def intent_heuristic_scores(subject: str | None, body: str | None) -> Dict[str, float]:
    text = compose_text(subject, body)
    lowered = normalize_text(text)
    scores: Dict[str, float] = {label: 0.01 for label in INTENT_KEYWORDS}
    for label, keywords in INTENT_KEYWORDS.items():
        hits = count_keyword_hits(lowered, keywords)
        if hits:
            scores[label] += min(0.85, hits * 0.18)
    if "?" in text:
        scores["information_request"] += 0.10
    if any(marker in lowered for marker in ("thank you", "thanks", "appreciate")):
        scores["thank_you"] += 0.25
    if any(marker in lowered for marker in ("meeting", "calendar", "availability")):
        scores["scheduling"] += 0.20
    total = sum(scores.values())
    if total > 0:
        scores = {label: value / total for label, value in scores.items()}
    return scores


def predict_intent_rule(subject: str | None, body: str | None) -> Tuple[str, float]:
    scores = intent_heuristic_scores(subject, body)
    label = max(scores, key=scores.get)
    confidence = scores[label]
    if confidence < 0.20:
        return DEFAULT_INTENT, 0.35
    return label, confidence


def weak_label_spam(subject: str | None, body: str | None) -> int:
    return int(spam_heuristic_score(subject, body) >= 0.55)


def promotional_score(sender: str | None, subject: str | None, body: str | None) -> int:
    """Return an integer score counting promotional signals in the message."""
    text = compose_text(subject, body)
    lowered = normalize_text(text)
    if not lowered:
        return 0
    score = 0
    # promo keyword hits
    score += count_keyword_hits(lowered, PROMO_KEYWORDS)
    # percent patterns
    if re.search(r"\b\d{1,3}%\b", lowered):
        score += 1
    # currency symbols
    if re.search(r"[$€£]", lowered):
        score += 1
    # urls
    if re.search(r"https?://|www\.", lowered):
        score += 1
    return score


def is_promotional(sender: str | None, subject: str | None, body: str | None, threshold: int = 2) -> bool:
    """Heuristic: message is promotional if multiple promo signals present and sender is not high-importance."""
    score = promotional_score(sender, subject, body)
    if score < threshold:
        return False
    # prefer not to flag messages from known important senders
    importance = sender_importance_score(sender)
    if importance >= 0.7:
        return False
    return True


def weak_label_reply_needed(sender: str | None, subject: str | None, body: str | None) -> int:
    return int(reply_needed_heuristic_score(sender, subject, body) >= 0.55)


def weak_label_intent(subject: str | None, body: str | None) -> str:
    label, _ = predict_intent_rule(subject, body)
    return label


def sender_importance_score(sender: str | None) -> float:
    if not sender:
        return 0.5
    lowered = sender.lower()
    if any(marker in lowered for marker in ("noreply", "no-reply", "newsletter", "marketing")):
        return 0.2
    if any(marker in lowered for marker in ("support", "help", "sales", "hr", "recruit")):
        return 0.8
    return 0.6
