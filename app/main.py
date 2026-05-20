from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router

from app.services.ollama_service import (
    OllamaClient,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.ollama = OllamaClient()

    yield


app = FastAPI(
    title="Email Intelligence API",
    version="3.0.0",
    lifespan=lifespan,
)

app.include_router(router)