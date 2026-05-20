"""Reusable helpers for training TF-IDF-based email classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None


def build_vectorizer(max_features: int = 50000) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )


def build_binary_model(name: str):
    name = name.lower()
    if name == "nb":
        return MultinomialNB()
    if name == "svm":
        return SVC(kernel="linear", probability=True, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    if name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available")
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        )
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def build_multiclass_model(name: str):
    name = name.lower()
    if name == "svm":
        return SVC(kernel="linear", probability=True, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    if name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available")
        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        )
    return LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto")


def tune_estimator(estimator: Any, model_name: str, X_train, y_train, scoring: str):
    model_name = model_name.lower()
    if model_name == "nb":
        param_grid = {"alpha": [0.25, 0.5, 1.0]}
    elif model_name == "svm":
        param_grid = {"C": [0.5, 1.0, 2.0]}
    elif model_name == "rf":
        param_grid = {"n_estimators": [150, 250], "max_depth": [None, 30]}
    elif model_name == "xgb":
        param_grid = {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
    else:
        param_grid = {"C": [0.5, 1.0, 2.0]}
    search = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_predictions(y_true, y_pred) -> Dict[str, Any]:
    labels = sorted(set(y_true))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def save_artifacts(model: Any, vectorizer: TfidfVectorizer, model_path: str, vectorizer_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def save_artifacts_with_scaler(model: Any, vectorizer: TfidfVectorizer, scaler: Any, model_path: str, vectorizer_path: str, scaler_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(scaler, scaler_path)
