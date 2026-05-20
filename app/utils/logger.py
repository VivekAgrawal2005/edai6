"""
==========================================================
Logger — Structured Logging Configuration
==========================================================
Centralized logging for requests, predictions, errors,
and processing time tracking.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "email_intelligence",
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str
        Path to the log file.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # --- Formatter ---
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")

    return logger


# ---------------------------------------------------------------------------
# Global application logger
# ---------------------------------------------------------------------------
logger = setup_logger()


def log_request(endpoint: str, email_id: str | None = None) -> None:
    """Log an incoming API request."""
    logger.info(f"REQUEST  | endpoint={endpoint} | email_id={email_id}")


def log_prediction(
    email_id: str | None,
    prediction_type: str,
    result: str,
    confidence: float,
) -> None:
    """Log a model prediction."""
    logger.info(
        f"PREDICT  | email_id={email_id} | type={prediction_type} "
        f"| result={result} | confidence={confidence:.4f}"
    )


def log_error(endpoint: str, error: str, email_id: str | None = None) -> None:
    """Log an error."""
    logger.error(f"ERROR    | endpoint={endpoint} | email_id={email_id} | {error}")


def log_processing_time(endpoint: str, time_ms: float, email_id: str | None = None) -> None:
    """Log processing time for a request."""
    logger.info(
        f"TIMING   | endpoint={endpoint} | email_id={email_id} | time={time_ms:.2f}ms"
    )
