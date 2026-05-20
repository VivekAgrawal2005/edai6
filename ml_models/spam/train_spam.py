"""Train a spam classifier from the Enron email corpus using weak labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split

from app.config import settings
from ml_models.shared import load_dataset, engineered_features_from_texts
from ml_models.train_utils import build_binary_model, build_vectorizer, evaluate_predictions, save_artifacts_with_scaler, tune_estimator
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spam classifier")
    parser.add_argument("--dataset", default=settings.dataset_path)
    parser.add_argument("--model", default="logreg", choices=["logreg", "nb", "svm", "rf", "xgb"])
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(settings.spam_model_path).parent))
    args = parser.parse_args()

    frame = load_dataset(max_rows=args.max_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["spam_label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["spam_label"],
    )

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # engineered numeric features
    X_train_feat = engineered_features_from_texts(X_train)
    X_test_feat = engineered_features_from_texts(X_test)
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    X_test_feat_scaled = scaler.transform(X_test_feat)

    # combine sparse TF-IDF with dense numeric features
    X_train_combined = hstack([X_train_vec, csr_matrix(X_train_feat_scaled)])
    X_test_combined = hstack([X_test_vec, csr_matrix(X_test_feat_scaled)])

    model = build_binary_model(args.model)
    best_params = None
    if args.tune:
        scoring = "f1"
        model, best_params = tune_estimator(model, args.model, X_train_combined, y_train, scoring)

    model.fit(X_train_combined, y_train)
    predictions = model.predict(X_test_combined)
    metrics = evaluate_predictions(y_test, predictions)
    if best_params:
        metrics["best_params"] = best_params

    output_dir = Path(args.output_dir)
    # save model, vectorizer, and scaler
    save_artifacts_with_scaler(model, vectorizer, scaler, str(output_dir / "spam_model.pkl"), str(output_dir / "vectorizer.pkl"), str(output_dir / "feature_scaler.pkl"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
