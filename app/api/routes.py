from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import (
    EmailInput,
)

from app.services.ollama_service import (
    OllamaClient,
)

router = APIRouter()


# =====================================================
# GET OLLAMA CLIENT
# =====================================================

def get_ollama(request: Request):

    ollama = getattr(
        request.app.state,
        "ollama",
        None,
    )

    if ollama is None:

        ollama = OllamaClient()

        request.app.state.ollama = ollama

    return ollama


# =====================================================
# HEALTH
# =====================================================

@router.get("/health")
async def health():

    return {
        "status": "healthy"
    }


# =====================================================
# MAIN EMAIL ANALYSIS
# =====================================================

@router.post("/analyze-email")
async def analyze_email(
    payload: EmailInput,
    request: Request,
):

    ollama = get_ollama(request)

    sender = (payload.sender or "").lower()

    # =================================================
    # SPECIAL TPO / NO-REPLY RULES
    # =================================================

    TPO_KEYWORDS = [
        "noreply_tpo",
        "tpo",
        "placement",
        "internship",
        "career",
    ]

    NO_REPLY_SENDERS = [
        "noreply",
        "no-reply",
        "donotreply",
    ]

    IMPORTANT_DOMAINS = [
        "tpo",
        "placement",
        "college.edu",
        "vit.edu",
    ]

    if (
        any(k in sender for k in TPO_KEYWORDS)
        or any(k in sender for k in NO_REPLY_SENDERS)
        or any(k in sender for k in IMPORTANT_DOMAINS)
    ):

        summary_result = await ollama.generate_tpo_summary(
            subject=payload.subject or "",
            body=payload.body or "",
        )

        return {
            "email_id": payload.email_id,
            "category": "ignorable",
            "summary": summary_result,
        }

    # =================================================
    # NORMAL EMAIL ANALYSIS
    # =================================================

    result = await ollama.analyze_email_complete(
        subject=payload.subject or "",
        body=payload.body or "",
    )

    result["email_id"] = payload.email_id

    return result