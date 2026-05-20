"""
==========================================================
Pydantic Schemas — Request & Response Models
==========================================================
Defines all input/output schemas for the API endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone


# ===================================================================
# REQUEST MODELS
# ===================================================================

class EmailInput(BaseModel):
    """Schema for a single email input."""
    email_id: Optional[str] = Field(None, description="Unique identifier for the email")
    sender: Optional[str] = Field(None, description="Sender email address")
    subject: Optional[str] = Field("", description="Email subject line")
    body: Optional[str] = Field("", description="Email body text")
    timestamp: Optional[str] = Field(None, description="Email timestamp in ISO format")


class BatchEmailInput(BaseModel):
    """Schema for batch email processing."""
    emails: List[EmailInput] = Field(..., description="List of emails to process")


class ReplyGenerationInput(BaseModel):
    """Schema for standalone reply generation."""
    intent: str = Field(..., description="Detected intent category")
    sender: Optional[str] = Field(None, description="Sender name or email")
    subject: Optional[str] = Field("", description="Email subject")
    body: Optional[str] = Field("", description="Email body")


# ===================================================================
# RESPONSE MODELS
# ===================================================================

class AnalysisResponse(BaseModel):
    """Full analysis response for a single email."""
    email_id: Optional[str] = None
    spam: bool
    spam_label: str = "ham"
    spam_confidence: float
    spam_reasons: List[str] = Field(default_factory=list)
    reply_needed: bool
    reply_confidence: float
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    generated_reply: Optional[str] = None
    processing_time_ms: float


class SpamResponse(BaseModel):
    """Response for spam-only prediction."""
    email_id: Optional[str] = None
    spam: bool
    spam_label: str = "ham"
    spam_confidence: float
    spam_reasons: List[str] = Field(default_factory=list)
    processing_time_ms: float


class ReplyNeededResponse(BaseModel):
    """Response for reply-needed-only prediction."""
    email_id: Optional[str] = None
    reply_needed: bool
    reply_confidence: float
    processing_time_ms: float


class IntentResponse(BaseModel):
    """Response for intent-only prediction."""
    email_id: Optional[str] = None
    intent: str
    intent_confidence: float
    processing_time_ms: float


class ReplyGenerationResponse(BaseModel):
    """Response for reply generation."""
    intent: str
    generated_reply: Optional[str] = None
    processing_time_ms: float


class BatchAnalysisResponse(BaseModel):
    """Response for batch email processing."""
    results: List[AnalysisResponse]
    total_emails: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: bool = True
    database_connected: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
