"""
Layer 2: Feedback Collection and Analysis
==========================================
Captures both implicit and explicit feedback signals from every query
processed by Layer 1 and stores them in a lightweight SQLite database
for consumption by Layer 3's Adaptation Engine.

Implicit signals (auto-captured):
  - Query reformulation events
  - Number of retrieval cycles required
  - Answer rejection / low-consistency events
  - Latency per query

Explicit signals (user-provided):
  - Thumbs up / thumbs down on a response
  - Free-text correction to a wrong answer
  - Relevance rating (0–5) on a retrieved passage
"""

from __future__ import annotations
import json
import logging
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.layers.layer1_self_repair import FeedbackEntry

logger = logging.getLogger(__name__)


class FeedbackDB:
    """
    SQLite-backed store for all feedback signals.
    Thread-safe for single-process use; use WAL mode for concurrent access.
    """

    # SQL schema
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS feedback (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id          TEXT    NOT NULL,
        timestamp         REAL    NOT NULL,
        original_query    TEXT,
        final_query       TEXT,
        retrieved_doc_ids TEXT,          -- JSON array
        relevance_score   REAL,
        consistency_score REAL,
        cycles_used       INTEGER,
        event_type        TEXT,          -- 'success' | 'low_relevance' | 'low_consistency'
        latency_ms        REAL,
        user_rating       INTEGER,       -- 1-5, NULL if not provided
        user_correction   TEXT,          -- free-text correction, NULL if not provided
        passage_rating    TEXT,          -- JSON {doc_id: rating}, NULL if not provided
        query_category    TEXT           -- set by Adaptation Engine after clustering
    );

    CREATE INDEX IF NOT EXISTS idx_timestamp    ON feedback(timestamp);
    CREATE INDEX IF NOT EXISTS idx_event_type   ON feedback(event_type);
    CREATE INDEX IF NOT EXISTS idx_query_category ON feedback(query_category);
    """

    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)
            # Enable WAL for better concurrent read performance
            conn.execute("PRAGMA journal_mode=WAL;")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    # ── Implicit feedback (auto from Layer 1) ──────────────────────────

    def log_implicit(self, entry: FeedbackEntry):
        """Store all implicit signals from a Layer 1 FeedbackEntry."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO feedback
                   (query_id, timestamp, original_query, final_query,
                    retrieved_doc_ids, relevance_score, consistency_score,
                    cycles_used, event_type, latency_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.query_id,
                    entry.timestamp,
                    entry.original_query,
                    entry.final_query,
                    json.dumps(entry.retrieved_doc_ids),
                    entry.relevance_score,
                    entry.consistency_score,
                    entry.cycles_used,
                    entry.event_type,
                    entry.latency_ms,
                ),
            )
        logger.debug(f"[Layer2] Logged implicit feedback for query_id={entry.query_id}")

    # ── Explicit feedback (user-provided) ──────────────────────────────

    def log_explicit_rating(self, query_id: str, rating: int):
        """
        User rates the response 1–5.
        Ratings ≤ 2 are treated as implicit rejection signals by Layer 3.
        """
        assert 1 <= rating <= 5, "Rating must be between 1 and 5"
        with self._conn() as conn:
            conn.execute(
                "UPDATE feedback SET user_rating=? WHERE query_id=?",
                (rating, query_id),
            )
        logger.info(f"[Layer2] Explicit rating={rating} for query_id={query_id}")

    def log_explicit_correction(self, query_id: str, correction: str):
        """User provides a free-text correction to a wrong answer."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE feedback SET user_correction=? WHERE query_id=?",
                (correction, query_id),
            )
        logger.info(f"[Layer2] Correction logged for query_id={query_id}")

    def log_passage_rating(self, query_id: str, ratings: dict[str, int]):
        """
        User rates individual retrieved passages.
        ratings = {doc_id: 0|1}  (0=irrelevant, 1=relevant)
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE feedback SET passage_rating=? WHERE query_id=?",
                (json.dumps(ratings), query_id),
            )

    # ── Query helpers for Layer 3 ──────────────────────────────────────

    def get_feedback_since(self, since_timestamp: float) -> list[dict]:
        """Return all feedback records after a given UNIX timestamp."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM feedback WHERE timestamp > ? ORDER BY timestamp",
                (since_timestamp,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_failure_patterns(self, since_timestamp: float) -> dict:
        """
        Aggregate failure statistics since a given time.
        Returns a dict Layer 3 uses to decide what to adapt.
        """
        rows = self.get_feedback_since(since_timestamp)
        if not rows:
            return {}

        total            = len(rows)
        low_rel_count    = sum(1 for r in rows if r["event_type"] == "low_relevance")
        low_cons_count   = sum(1 for r in rows if r["event_type"] == "low_consistency")
        multi_cycle      = sum(1 for r in rows if r["cycles_used"] > 1)
        avg_rel_score    = sum(r["relevance_score"]   for r in rows) / total
        avg_cons_score   = sum(r["consistency_score"] for r in rows) / total
        explicit_corrections = [r for r in rows if r["user_correction"]]

        # Collect corrections for query expansion vocabulary
        corrections_text = [r["user_correction"] for r in explicit_corrections
                            if r["user_correction"]]

        return {
            "total_queries":          total,
            "low_relevance_rate":     low_rel_count / total,
            "low_consistency_rate":   low_cons_count / total,
            "multi_cycle_rate":       multi_cycle / total,
            "avg_relevance_score":    avg_rel_score,
            "avg_consistency_score":  avg_cons_score,
            "explicit_corrections":   corrections_text,
            "needs_retrieval_update": (avg_rel_score < 0.65),
            "needs_consistency_update": (avg_cons_score < 0.72),
        }

    def get_total_query_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    def set_query_category(self, query_id: str, category: str):
        """Used by Layer 3 to annotate queries with their cluster label."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE feedback SET query_category=? WHERE query_id=?",
                (category, query_id),
            )


class FeedbackCollector:
    """
    High-level interface used by the main pipeline.
    Wraps FeedbackDB and adds pattern analysis logic.
    """

    def __init__(self, db_path: str = "data/feedback.db"):
        self.db = FeedbackDB(db_path)
        self._last_cycle_timestamp = time.time()

    def record(self, entry: FeedbackEntry):
        """Called automatically by the pipeline after every query."""
        self.db.log_implicit(entry)

    def user_rates(self, query_id: str, rating: int):
        """Call this from your UI when a user rates a response."""
        self.db.log_explicit_rating(query_id, rating)

    def user_corrects(self, query_id: str, correction: str):
        """Call this from your UI when a user provides a correction."""
        self.db.log_explicit_correction(query_id, correction)

    def user_rates_passage(self, query_id: str, ratings: dict[str, int]):
        """Call this from your UI when a user rates individual passages."""
        self.db.log_passage_rating(query_id, ratings)

    def get_patterns_for_adaptation(self) -> dict:
        """Layer 3 calls this to get aggregated failure patterns."""
        patterns = self.db.get_failure_patterns(self._last_cycle_timestamp)
        self._last_cycle_timestamp = time.time()
        return patterns

    def total_queries(self) -> int:
        return self.db.get_total_query_count()
