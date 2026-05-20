"""Validate model artifacts by loading via ModelRegistry and benchmarking inference."""

from __future__ import annotations

import json
import time
from statistics import mean
from pathlib import Path

from app.services.model_service import ModelRegistry


OUTPUT = Path(__file__).with_name("validation_report.json")


def time_call(func, *args, repeats: int = 50, warmup: int = 5):
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "runs": repeats,
        "mean_ms": mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main() -> None:
    registry = ModelRegistry()

    report = {"models": {}, "summary": {}}

    # Check availability
    report["models"]["spam"] = {"available": registry.spam.available}
    report["models"]["reply_needed"] = {"available": registry.reply_needed.available}
    report["models"]["intent"] = {"available": registry.intent.available}

    # Sample inputs
    subject = "Meeting request: project sync"
    body = "Hi team, can we schedule a 30 minute call to review the project status next week? Thanks."

    # Benchmark spam
    if registry.spam.available:
        spam_t = time_call(lambda: registry.spam.predict(subject, body), repeats=100, warmup=5)
        report["models"]["spam"]["inference_ms"] = spam_t
        # call once for result
        sp = registry.spam.predict(subject, body)
        report["models"]["spam"]["sample_result"] = {"label": sp.label, "confidence": sp.confidence}
        # verify scaler artifact is present
        try:
            from app.config import settings
            from pathlib import Path
            report["models"]["spam"]["scaler_path_exists"] = Path(settings.spam_feature_scaler_path).exists()
        except Exception:
            report["models"]["spam"]["scaler_path_exists"] = False

    # Benchmark reply_needed
    if registry.reply_needed.available:
        reply_t = time_call(lambda: registry.reply_needed.predict("someone@example.com", subject, body), repeats=100, warmup=5)
        report["models"]["reply_needed"]["inference_ms"] = reply_t
        rp = registry.reply_needed.predict("someone@example.com", subject, body)
        report["models"]["reply_needed"]["sample_result"] = {"label": rp.label, "confidence": rp.confidence}

    # Benchmark intent
    if registry.intent.available:
        intent_t = time_call(lambda: registry.intent.predict(subject, body), repeats=100, warmup=5)
        report["models"]["intent"]["inference_ms"] = intent_t
        it = registry.intent.predict(subject, body)
        report["models"]["intent"]["sample_result"] = {"label": it.label, "confidence": it.confidence}

    report["summary"]["total_models_available"] = sum(1 for m in report["models"].values() if m.get("available"))

    with open(OUTPUT, "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
    
