"""Train a reply-needed classifier from weak labels derived from Enron emails."""

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
from ml_models.shared import load_dataset
from ml_models.train_utils import build_binary_model, build_vectorizer, evaluate_predictions, save_artifacts, tune_estimator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reply-needed classifier")
    parser.add_argument("--dataset", default=settings.dataset_path)
    parser.add_argument("--model", default="xgb", choices=["logreg", "nb", "svm", "rf", "xgb"])
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(settings.reply_model_path).parent))
    args = parser.parse_args()

    frame = load_dataset(max_rows=args.max_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["reply_label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["reply_label"],
    )

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = build_binary_model(args.model)
    best_params = None
    if args.tune:
        model, best_params = tune_estimator(model, args.model, X_train_vec, y_train, "f1")

    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    metrics = evaluate_predictions(y_test, predictions)
    if best_params:
        metrics["best_params"] = best_params

    output_dir = Path(args.output_dir)
    save_artifacts(model, vectorizer, str(output_dir / "reply_model.pkl"), str(output_dir / "vectorizer.pkl"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
