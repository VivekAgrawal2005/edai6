"""Application entrypoint for the Email Intelligence API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.database.db import init_db
from app.preprocessing.text_cleaner import download_nltk_data
from app.services.email_processor import EmailProcessor
from app.services.model_service import ModelRegistry
from app.services.reply.reply_service import ReplyGenerator
from app.services.rules.rules_service import RuleEngine
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    download_nltk_data()
    init_db()
    app.state.processor = EmailProcessor(ModelRegistry(), ReplyGenerator(), RuleEngine())
    logger.info("Email Intelligence API started")
    yield
    logger.info("Email Intelligence API stopped")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

app.include_router(router)
