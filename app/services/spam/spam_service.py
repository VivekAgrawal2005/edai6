import logging
import re

from dataclasses import dataclass
from typing import Dict, List

from transformers import pipeline


logger = logging.getLogger(__name__)


@dataclass
class SpamDecision:
    spam: bool
    spam_label: str
    spam_confidence: float
    spam_reasons: List[str]


class SpamClassifier:

    def __init__(self):

        self.classifier = None
        self.model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

        self.SPAM_KEYWORDS = [
            "free",
            "winner",
            "won",
            "gift",
            "claim",
            "reward",
            "click here",
            "urgent",
            "limited offer",
            "voucher",
            "congratulations",
            "buy now",
            "discount",
            "offer expires",
            "earn money",
            "crypto",
            "investment opportunity"
        ]

        self.PHISHING_KEYWORDS = [
            "verify your account",
            "verify your password",
            "bank account suspended",
            "click here immediately",
            "urgent action required",
            "confirm your identity",
            "reset your password",
            "security alert",
            "unauthorized login",
            "account locked",
            "restore access",
            "payment failed",
            "login attempt",
            "suspicious activity",
            "verify your banking details",
            "your account has been suspended"
        ]

        self.URGENCY_WORDS = [
            "urgent",
            "immediately",
            "action required",
            "now",
            "limited time",
            "as soon as possible"
        ]

        self.BANKING_WORDS = [
            "bank",
            "credit card",
            "account",
            "transaction",
            "payment",
            "debit",
            "upi",
            "wallet"
        ]

        self.SUSPICIOUS_DOMAINS = [
            "verify",
            "alert",
            "secure-login",
            "banking-alert",
            "account-update",
            "claim-reward",
            "bonus",
            "crypto",
            "free-money"
        ]

        self.load_model()

    def load_model(self):

        try:

            logger.info("Loading Hugging Face spam classifier...")

            self.classifier = pipeline(
                "text-classification",
                model=self.model_name
            )

            logger.info("Spam classifier loaded successfully")

        except Exception as e:

            logger.error(f"Failed to load HF model: {e}")
            self.classifier = None

    def _calculate_hf_score(self, text: str) -> float:

        if not self.classifier:
            return 0.0

        try:

            logger.info("Using HuggingFace spam classifier")

            prediction = self.classifier(text)[0]

            label = prediction["label"].lower()
            score = prediction["score"]

            if label == "spam":
                return score

            return 1 - score

        except Exception as e:

            logger.error(f"HF prediction failed: {e}")
            return 0.0

    def _detect_urls(self, text: str) -> List[str]:

        return re.findall(r'https?://\S+', text)

    def predict(
        self,
        sender: str,
        subject: str,
        body: str
    ) -> SpamDecision:

        text = f"{subject} {body}".lower()

        spam_score = 0.0
        phishing_score = 0.0

        reasons = []

        # ----------------------------------------
        # HuggingFace Score
        # ----------------------------------------

        hf_score = self._calculate_hf_score(text)

        spam_score += hf_score * 0.5

        # ----------------------------------------
        # Spam Keywords
        # ----------------------------------------

        spam_keyword_hits = 0

        for keyword in self.SPAM_KEYWORDS:

            if keyword in text:

                spam_keyword_hits += 1
                spam_score += 0.08

        if spam_keyword_hits > 0:

            reasons.append("Contains spam keywords")

        # ----------------------------------------
        # Phishing Keywords
        # ----------------------------------------

        phishing_hits = 0

        for keyword in self.PHISHING_KEYWORDS:

            if keyword in text:

                phishing_hits += 1
                phishing_score += 0.25

        if phishing_hits > 0:

            reasons.append("Contains phishing-style language")

        # ----------------------------------------
        # Suspicious Sender
        # ----------------------------------------

        sender_lower = sender.lower()

        if any(x in sender_lower for x in self.SUSPICIOUS_DOMAINS):

            phishing_score += 0.2
            reasons.append("Suspicious sender domain")

        # ----------------------------------------
        # URL Detection
        # ----------------------------------------

        urls = self._detect_urls(text)

        if len(urls) > 0:

            phishing_score += 0.15
            reasons.append("Contains suspicious URLs")

        # ----------------------------------------
        # Urgency Detection
        # ----------------------------------------

        urgency_hits = 0

        for word in self.URGENCY_WORDS:

            if word in text:

                urgency_hits += 1

                spam_score += 0.05
                phishing_score += 0.05

        if urgency_hits > 0:

            reasons.append("Excessive urgency language")

        # ----------------------------------------
        # Banking Detection
        # ----------------------------------------

        banking_hits = 0

        for word in self.BANKING_WORDS:

            if word in text:

                banking_hits += 1
                phishing_score += 0.08

        if banking_hits > 0:

            reasons.append("Financial account impersonation")

        # ----------------------------------------
        # ALL CAPS Subject
        # ----------------------------------------

        if subject.isupper() and len(subject) > 8:

            spam_score += 0.1
            reasons.append("ALL CAPS subject")

        # ----------------------------------------
        # Excessive Exclamation Marks
        # ----------------------------------------

        if text.count("!") > 3:

            spam_score += 0.1
            reasons.append("Too many exclamation marks")

        # ----------------------------------------
        # Final Scores
        # ----------------------------------------

        spam_score = min(round(spam_score, 4), 0.98)
        phishing_score = min(round(phishing_score, 4), 0.98)

        # ----------------------------------------
        # Final Classification
        # ----------------------------------------

        if phishing_score >= 0.5:

            final_label = "phishing"
            final_confidence = phishing_score
            is_spam = True

        elif spam_score >= 0.75:

            final_label = "spam"
            final_confidence = spam_score
            is_spam = True

        elif spam_score >= 0.45:

            final_label = "suspicious"
            final_confidence = spam_score
            is_spam = True

        else:

            final_label = "ham"
            final_confidence = spam_score
            is_spam = False

        return SpamDecision(
            spam=is_spam,
            spam_label=final_label,
            spam_confidence=final_confidence,
            spam_reasons=list(set(reasons))
        )