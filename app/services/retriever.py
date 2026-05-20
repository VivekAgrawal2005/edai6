"""Simple retrieval augmentation using TF-IDF cosine similarity over historical replies.

This provides example replies to include in prompts sent to Ollama.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.database import db
from app.config import settings
from app.utils.logger import logger


class Retriever:
    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.rag_top_k
        self._vectorizer: TfidfVectorizer | None = None
        self._corpus: List[str] = []
        self._matrix = None

    def _load_corpus(self) -> None:
        preds = db.get_all_predictions(limit=1000)
        replies = [p.get("generated_reply") for p in preds if p.get("generated_reply")]
        # fallback: no replies
        self._corpus = replies or []
        if self._corpus:
            self._vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            self._matrix = self._vectorizer.fit_transform(self._corpus)

    def retrieve(self, subject: str | None, body: str | None, k: int | None = None) -> List[str]:
        if not self._corpus:
            try:
                self._load_corpus()
            except Exception as exc:
                logger.warning(f"Retriever load failed: {exc}")
                return []
        if not self._corpus or self._matrix is None or self._vectorizer is None:
            return []
        query = f"{subject or ''} {body or ''}"
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        top_k = k or self.top_k
        idx = np.argsort(sims)[::-1][:top_k]
        results = [self._corpus[i] for i in idx if sims[i] > 0.1]
        return results


__all__ = ["Retriever"]
