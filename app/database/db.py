"""
==========================================================
Database — SQLite Integration
==========================================================
SQLite database for storing emails, predictions,
and processing logs.
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterator

from app.config import settings
from app.utils.logger import logger

DB_PATH = settings.database_path


# ===================================================================
# Schema Definitions
# ===================================================================

CREATE_EMAILS_TABLE = """
CREATE TABLE IF NOT EXISTS emails (
    id              TEXT PRIMARY KEY,
    sender          TEXT,
    subject         TEXT,
    body            TEXT,
    timestamp       TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
"""

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id            TEXT,
    spam                INTEGER,
    spam_confidence     REAL,
    reply_needed        INTEGER,
    reply_confidence    REAL,
    intent              TEXT,
    intent_confidence   REAL,
    generated_reply     TEXT,
    processing_time_ms  REAL,
    created_at          TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (email_id) REFERENCES emails(id)
);
"""

CREATE_PROCESSING_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS processing_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id            TEXT,
    endpoint            TEXT,
    status              TEXT,
    message             TEXT,
    processing_time_ms  REAL,
    created_at          TEXT DEFAULT (datetime('now'))
);
"""


# ===================================================================
# Database Initialization
# ===================================================================

def init_db() -> None:
    """Create the database and tables if they don't exist."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_EMAILS_TABLE)
            cursor.execute(CREATE_PREDICTIONS_TABLE)
            cursor.execute(CREATE_PROCESSING_LOGS_TABLE)
            conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()


# ===================================================================
# CRUD Operations
# ===================================================================

def store_email(
    email_id: str,
    sender: Optional[str],
    subject: Optional[str],
    body: Optional[str],
    timestamp: Optional[str],
) -> None:
    """Insert an email record into the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO emails (id, sender, subject, body, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email_id, sender, subject, body, timestamp),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store email {email_id}: {e}")


def store_prediction(
    email_id: str,
    spam: bool,
    spam_confidence: float,
    reply_needed: bool,
    reply_confidence: float,
    intent: Optional[str],
    intent_confidence: float,
    generated_reply: Optional[str],
    processing_time_ms: float,
) -> None:
    """Insert a prediction record into the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions
                (email_id, spam, spam_confidence, reply_needed, reply_confidence,
                 intent, intent_confidence, generated_reply, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email_id,
                    int(spam),
                    spam_confidence,
                    int(reply_needed),
                    reply_confidence,
                    intent,
                    intent_confidence,
                    generated_reply,
                    processing_time_ms,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store prediction for {email_id}: {e}")


def store_processing_log(
    endpoint: str,
    status: str,
    message: str,
    email_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> None:
    """Insert a processing log record into the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO processing_logs
                (email_id, endpoint, status, message, processing_time_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email_id, endpoint, status, message, processing_time_ms),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store processing log for {endpoint}: {e}")


def get_prediction(email_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the latest prediction for a given email_id."""
    try:
        with get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM predictions
                WHERE email_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (email_id,),
            )
            row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve prediction for {email_id}: {e}")
        return None


def get_all_predictions(limit: int = 100) -> list:
    """Retrieve recent predictions from the database."""
    try:
        with get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to retrieve predictions: {e}")
        return []


def check_db_connection() -> bool:
    """Check if the database is accessible."""
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
