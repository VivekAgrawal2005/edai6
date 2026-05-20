"""Model loading and prediction services for the email intelligence backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from app.config import settings
from app.preprocessing.text_cleaner import clean_text
from app.services.heuristics import (
    compose_text,
    predict_intent_rule,
    reply_needed_heuristic_score,
    spam_heuristic_score,
)
from app.services.spam_service import SpamClassifier as HybridSpamClassifier
from ml_models.shared import engineered_features_from_texts
from scipy.sparse import hstack, csr_matrix


def _safe_load(path: str) -> Any | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return joblib.load(candidate)


def _probability_from_scores(scores: np.ndarray, classes: list[str], target_label: str | None = None) -> Tuple[Any, float]:
    if scores.ndim != 1:
        scores = scores.ravel()
    index = int(np.argmax(scores))
    label = classes[index]
    confidence = float(scores[index])
    if target_label is not None and target_label in classes:
        target_index = classes.index(target_label)
        return target_label, float(scores[target_index])
    return label, confidence


@dataclass
class BinaryPrediction:
    label: bool
    confidence: float


@dataclass
class MultiClassPrediction:
    label: str
    confidence: float


class LoadedTextClassifier:
    def __init__(self, model_path: str, vectorizer_path: str, positive_label: int | str = 1):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.positive_label = positive_label
        self.model = _safe_load(model_path)
        self.vectorizer = _safe_load(vectorizer_path)

    @property
    def available(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict_binary(self, text: str, heuristic_score: float) -> BinaryPrediction:
        if self.available:
            features = self.vectorizer.transform([text])
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                classes = list(self.model.classes_)
                if self.positive_label in classes:
                    positive_index = classes.index(self.positive_label)
                    confidence = float(probabilities[positive_index])
                    return BinaryPrediction(confidence >= 0.5, confidence)
                positive_index = 1 if len(probabilities) > 1 else 0
                confidence = float(probabilities[positive_index])
                return BinaryPrediction(confidence >= 0.5, confidence)
            prediction = self.model.predict(features)[0]
            confidence = 0.85
            return BinaryPrediction(bool(prediction), confidence)

        confidence = float(max(heuristic_score, 1.0 - heuristic_score))
        return BinaryPrediction(heuristic_score >= 0.5, confidence)


class SpamClassifier(LoadedTextClassifier):
    def __init__(self) -> None:
        super().__init__(settings.spam_model_path, settings.spam_vectorizer_path)
        self.scaler = _safe_load(settings.spam_feature_scaler_path)
        self._hf_classifier = None
        self._hf_init_attempted = False

    def _ensure_hf_classifier(self) -> None:
        if self._hf_init_attempted:
            return
        self._hf_init_attempted = True
        if not settings.use_hf_spam_classifier:
            return
        try:
            from transformers import pipeline

            self._hf_classifier = pipeline(
                "text-classification",
                model=settings.hf_spam_model_id,
                tokenizer=settings.hf_spam_model_id,
                device=settings.hf_spam_device,
                truncation=True,
            )
        except Exception:
            self._hf_classifier = None

    @staticmethod
    def _spam_probability_from_hf_result(result: Dict[str, Any]) -> float:
        label = str(result.get("label", "")).strip().lower()
        score = float(result.get("score", 0.5))
        if "spam" in label:
            return score
        if "ham" in label or "not_spam" in label or "non_spam" in label:
            return 1.0 - score
        if label in {"label_1", "1"}:
            return score
        if label in {"label_0", "0"}:
            return 1.0 - score
        # conservative fallback when label schema is unknown
        return score if score >= 0.5 else 1.0 - score

    def predict(self, subject: str | None, body: str | None) -> BinaryPrediction:
        raw_text = compose_text(subject, body)
        text = clean_text(raw_text)
        score = spam_heuristic_score(subject, body)
        self._ensure_hf_classifier()

        if self._hf_classifier is not None and raw_text:
            try:
                result = self._hf_classifier(raw_text[:4000])[0]
                spam_probability = self._spam_probability_from_hf_result(result)
                hf_prediction = BinaryPrediction(spam_probability >= settings.spam_threshold, spam_probability)
                if hf_prediction.confidence >= 0.50:
                    return hf_prediction
            except Exception:
                pass

        # if model and vectorizer available, combine TF-IDF with engineered numeric features
        if self.available:
            tfidf = self.vectorizer.transform([text])
            if self.scaler is not None:
                feats = engineered_features_from_texts([text])
                try:
                    feats_scaled = self.scaler.transform(feats)
                    combined = hstack([tfidf, csr_matrix(feats_scaled)])
                except Exception:
                    combined = tfidf
            else:
                combined = tfidf

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(combined)[0]
                classes = list(self.model.classes_)
                if self.positive_label in classes:
                    positive_index = classes.index(self.positive_label)
                    confidence = float(probs[positive_index])
                else:
                    positive_index = 1 if len(probs) > 1 else 0
                    confidence = float(probs[positive_index])
                pred = confidence >= 0.5
                prediction = BinaryPrediction(pred, confidence)
            else:
                pred = bool(self.model.predict(combined)[0])
                prediction = BinaryPrediction(pred, 0.85)
        else:
            prediction = self.predict_binary(text, score)
        if not self.available:
            return prediction
        return prediction if prediction.confidence >= settings.spam_threshold else BinaryPrediction(False, prediction.confidence)


class ReplyNeededClassifier(LoadedTextClassifier):
    def __init__(self) -> None:
        super().__init__(settings.reply_model_path, settings.reply_vectorizer_path)

    def predict(self, sender: str | None, subject: str | None, body: str | None) -> BinaryPrediction:
        text = clean_text(f"{subject or ''} {body or ''}")
        score = reply_needed_heuristic_score(sender, subject, body)
        prediction = self.predict_binary(text, score)
        if not self.available:
            return prediction
        return prediction if prediction.confidence >= settings.reply_threshold else BinaryPrediction(False, prediction.confidence)


class IntentClassifier:
    def __init__(self) -> None:
        self.model = _safe_load(settings.intent_model_path)
        self.vectorizer = _safe_load(settings.intent_vectorizer_path)

    @property
    def available(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, subject: str | None, body: str | None) -> MultiClassPrediction:
        text = clean_text(f"{subject or ''} {body or ''}")
        if self.available:
            features = self.vectorizer.transform([text])
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                classes = list(self.model.classes_)
                label, confidence = _probability_from_scores(np.asarray(probabilities), classes)
                return MultiClassPrediction(str(label), float(confidence))
            prediction = self.model.predict(features)[0]
            return MultiClassPrediction(str(prediction), 0.85)

        label, confidence = predict_intent_rule(subject, body)
        return MultiClassPrediction(label, confidence)


class ModelRegistry:
    """Loads models once and exposes typed prediction helpers."""

    def __init__(self) -> None:
        self.spam = HybridSpamClassifier()
        self.reply_needed = ReplyNeededClassifier()
        self.intent = IntentClassifier()
