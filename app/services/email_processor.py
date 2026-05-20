"""High-level orchestration for email analysis requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import List

from app.database.db import store_email, store_prediction, store_processing_log
from app.models.schemas import AnalysisResponse, BatchAnalysisResponse, EmailInput, IntentResponse, ReplyGenerationInput, ReplyGenerationResponse, ReplyNeededResponse, SpamResponse
from app.preprocessing.text_cleaner import clean_text
from app.services.heuristics import compose_text
from app.services.model_service import ModelRegistry
from app.services.reply.reply_service import ReplyGenerator
from app.services.rules.rules_service import RuleEngine
from app.utils.logger import log_error, log_prediction, log_processing_time, log_request, logger


@dataclass
class ProcessedEmail:
    email_id: str | None
    sender: str | None
    subject: str | None
    body: str | None
    timestamp: str | None


class EmailProcessor:
    def __init__(self, models: ModelRegistry, reply_generator: ReplyGenerator, rule_engine: RuleEngine) -> None:
        self.models = models
        self.reply_generator = reply_generator
        self.rule_engine = rule_engine

    def _store_request(self, email: EmailInput) -> None:
        if email.email_id:
            store_email(email.email_id, email.sender, email.subject, email.body, email.timestamp)

    def predict_spam(self, email: EmailInput) -> SpamResponse:
        result = self.models.spam.predict(email.sender, email.subject, email.body)
        return SpamResponse(
            email_id=email.email_id,
            spam=result.spam,
            spam_label=result.spam_label,
            spam_confidence=round(result.spam_confidence, 4),
            spam_reasons=result.spam_reasons,
            processing_time_ms=0.0,
        )

    def predict_reply_needed(self, email: EmailInput) -> ReplyNeededResponse:
        result = self.models.reply_needed.predict(email.sender, email.subject, email.body)
        return ReplyNeededResponse(
            email_id=email.email_id,
            reply_needed=result.label,
            reply_confidence=round(result.confidence, 4),
            processing_time_ms=0.0,
        )

    def predict_intent(self, email: EmailInput) -> IntentResponse:
        result = self.models.intent.predict(email.subject, email.body)
        return IntentResponse(
            email_id=email.email_id,
            intent=result.label,
            intent_confidence=round(result.confidence, 4),
            processing_time_ms=0.0,
        )

    def generate_reply(self, intent: str, email: EmailInput) -> ReplyGenerationResponse:
        reply = self.reply_generator.generate(intent, email.sender, email.subject, email.body, email.email_id)
        return ReplyGenerationResponse(intent=intent, generated_reply=reply, processing_time_ms=0.0)

    def analyze(self, email: EmailInput) -> AnalysisResponse:
        start = time.perf_counter()
        log_request("/analyze-email", email.email_id)
        try:
            self._store_request(email)

            spam_result = self.models.spam.predict(email.sender, email.subject, email.body)
            spam = spam_result.spam
            spam_label = spam_result.spam_label
            spam_confidence = round(spam_result.spam_confidence, 4)

            if spam_label != "ham":
                response = AnalysisResponse(
                    email_id=email.email_id,
                    spam=spam,
                    spam_label=spam_label,
                    spam_confidence=spam_confidence,
                    spam_reasons=spam_result.spam_reasons,
                    reply_needed=False,
                    reply_confidence=round(max(0.01, 1.0 - spam_confidence), 4),
                    intent=None,
                    intent_confidence=0.0,
                    generated_reply=None,
                    processing_time_ms=0.0,
                )
            else:
                reply_result = self.models.reply_needed.predict(email.sender, email.subject, email.body)
                intent_result = self.models.intent.predict(email.subject, email.body)
                generated_reply = None
                if self.rule_engine.should_generate_reply(False, reply_result.label, intent_result.label).allow_reply:
                    generated_reply = self.reply_generator.generate(
                        intent_result.label,
                        email.sender,
                        email.subject,
                        email.body,
                        email.email_id,
                    )
                generated_reply = self.rule_engine.apply(False, reply_result.label, intent_result.label, generated_reply)

                response = AnalysisResponse(
                    email_id=email.email_id,
                    spam=spam,
                    spam_label=spam_label,
                    spam_confidence=spam_confidence,
                    spam_reasons=spam_result.spam_reasons,
                    reply_needed=reply_result.label,
                    reply_confidence=round(reply_result.confidence, 4),
                    intent=intent_result.label,
                    intent_confidence=round(intent_result.confidence, 4),
                    generated_reply=generated_reply,
                    processing_time_ms=0.0,
                )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            response.processing_time_ms = round(elapsed_ms, 2)
            if email.email_id:
                store_prediction(
                    email.email_id,
                    response.spam,
                    response.spam_confidence,
                    response.reply_needed,
                    response.reply_confidence,
                    response.intent,
                    response.intent_confidence,
                    response.generated_reply,
                    response.processing_time_ms,
                )
            store_processing_log("/analyze-email", "success", "analysis completed", email.email_id, response.processing_time_ms)
            log_prediction(email.email_id, "spam", response.spam_label, response.spam_confidence)
            log_prediction(email.email_id, "reply_needed", str(response.reply_needed), response.reply_confidence)
            if response.intent:
                log_prediction(email.email_id, "intent", response.intent, response.intent_confidence)
            log_processing_time("/analyze-email", response.processing_time_ms, email.email_id)
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            store_processing_log("/analyze-email", "error", str(exc), email.email_id, elapsed_ms)
            log_error("/analyze-email", str(exc), email.email_id)
            raise

    async def analyze_async(self, email: EmailInput) -> AnalysisResponse:
        return await asyncio.to_thread(self.analyze, email)

    async def analyze_batch_async(self, emails: List[EmailInput]) -> BatchAnalysisResponse:
        started = time.perf_counter()
        results = await asyncio.gather(*(self.analyze_async(email) for email in emails))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return BatchAnalysisResponse(results=list(results), total_emails=len(results), total_processing_time_ms=round(elapsed_ms, 2))
