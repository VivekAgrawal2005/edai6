"""FastAPI routes for the email intelligence backend."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.database.db import check_db_connection
from app.models.schemas import AnalysisResponse, BatchAnalysisResponse, BatchEmailInput, EmailInput, HealthResponse, IntentResponse, ReplyGenerationInput, ReplyGenerationResponse, ReplyNeededResponse, SpamResponse
from app.services.email_processor import EmailProcessor


router = APIRouter()


def get_processor(request: Request) -> EmailProcessor:
    processor = getattr(request.app.state, "processor", None)
    if processor is None:
        raise HTTPException(status_code=503, detail="Model services are not initialized")
    return processor


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    processor = get_processor(request)
    return HealthResponse(models_loaded=processor is not None, database_connected=check_db_connection())


@router.post("/analyze-email", response_model=AnalysisResponse)
async def analyze_email(payload: EmailInput, request: Request) -> AnalysisResponse:
    processor = get_processor(request)
    return await processor.analyze_async(payload)


@router.post("/predict-spam", response_model=SpamResponse)
async def predict_spam(payload: EmailInput, request: Request) -> SpamResponse:
    processor = get_processor(request)
    result = processor.predict_spam(payload)
    return result


@router.post("/predict-reply-needed", response_model=ReplyNeededResponse)
async def predict_reply_needed(payload: EmailInput, request: Request) -> ReplyNeededResponse:
    processor = get_processor(request)
    return processor.predict_reply_needed(payload)


@router.post("/predict-intent", response_model=IntentResponse)
async def predict_intent(payload: EmailInput, request: Request) -> IntentResponse:
    processor = get_processor(request)
    return processor.predict_intent(payload)


@router.post("/generate-reply", response_model=ReplyGenerationResponse)
async def generate_reply(payload: ReplyGenerationInput, request: Request) -> ReplyGenerationResponse:
    processor = get_processor(request)
    reply = processor.reply_generator.generate(payload.intent, payload.sender, payload.subject, payload.body)
    return ReplyGenerationResponse(intent=payload.intent, generated_reply=reply, processing_time_ms=0.0)


@router.post("/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_batch(payload: BatchEmailInput, request: Request) -> BatchAnalysisResponse:
    processor = get_processor(request)
    emails = payload.emails
    return await processor.analyze_batch_async(emails)
