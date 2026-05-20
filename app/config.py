"""Application configuration for the Email Intelligence API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(env_value: str | None, default_relative: str) -> str:
    candidate = Path(env_value).expanduser() if env_value else PROJECT_ROOT / default_relative
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return str(candidate)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppSettings:
    api_title: str = os.getenv("API_TITLE", "Email Intelligence API")
    api_version: str = os.getenv("API_VERSION", "1.0.0")
    api_description: str = os.getenv(
        "API_DESCRIPTION",
        "Traditional NLP and machine learning backend for email analysis.",
    )
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./email_intelligence.db")
    spam_threshold: float = float(os.getenv("SPAM_THRESHOLD", "0.60"))
    reply_threshold: float = float(os.getenv("REPLY_THRESHOLD", "0.55"))
    intent_threshold: float = float(os.getenv("INTENT_THRESHOLD", "0.45"))
    dataset_path: str = _resolve_path(os.getenv("DATASET_PATH"), "dataset/emails.csv")
    templates_path: str = _resolve_path(os.getenv("TEMPLATES_PATH"), "templates/reply_templates.json")
    spam_model_path: str = _resolve_path(os.getenv("SPAM_MODEL_PATH"), "ml_models/spam/spam_model.pkl")
    spam_vectorizer_path: str = _resolve_path(os.getenv("SPAM_VECTORIZER_PATH"), "ml_models/spam/vectorizer.pkl")
    reply_model_path: str = _resolve_path(os.getenv("REPLY_MODEL_PATH"), "ml_models/reply_needed/reply_model.pkl")
    reply_vectorizer_path: str = _resolve_path(os.getenv("REPLY_VECTORIZER_PATH"), "ml_models/reply_needed/vectorizer.pkl")
    intent_model_path: str = _resolve_path(os.getenv("INTENT_MODEL_PATH"), "ml_models/intent/intent_model.pkl")
    intent_vectorizer_path: str = _resolve_path(os.getenv("INTENT_VECTORIZER_PATH"), "ml_models/intent/vectorizer.pkl")
    spam_feature_scaler_path: str = _resolve_path(os.getenv("SPAM_FEATURE_SCALER_PATH"), "ml_models/spam/feature_scaler.pkl")
    use_hf_spam_classifier: bool = _get_bool("USE_HF_SPAM_CLASSIFIER", True)
    hf_spam_model_id: str = os.getenv("HF_SPAM_MODEL_ID", "mrm8488/bert-tiny-finetuned-sms-spam-detection")
    hf_spam_device: int = int(os.getenv("HF_SPAM_DEVICE", "-1"))
    # Ollama (local LLM) settings
    use_ollama: bool = _get_bool("USE_OLLAMA", False)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_timeout_sec: int = int(os.getenv("OLLAMA_TIMEOUT_SEC", "15"))
    ollama_retries: int = int(os.getenv("OLLAMA_RETRIES", "2"))
    # Retrieval augmentation
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))

    @property
    def database_path(self) -> str:
        if self.database_url.startswith("sqlite:///"):
            return self.database_url.replace("sqlite:///", "", 1)
        return self.database_url


settings = AppSettings()
