"""
Tests for Layer 2 — Feedback Collection & Analysis
"""
import pytest
import tempfile
import time
from src.layers.layer1_self_repair import FeedbackEntry
from src.layers.layer2_feedback import FeedbackDB, FeedbackCollector


def make_entry(query_id="q1", cycles=1, event="success", s_rel=0.85, s_cons=0.90):
    return FeedbackEntry(
        query_id          = query_id,
        original_query    = "Test question?",
        final_query       = "Test question expanded?",
        retrieved_doc_ids = ["doc1", "doc2"],
        relevance_score   = s_rel,
        consistency_score = s_cons,
        cycles_used       = cycles,
        event_type        = event,
        latency_ms        = 320.0,
        timestamp         = time.time(),
    )


class TestFeedbackDB:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = FeedbackDB(db_path=f"{self.tmpdir}/test_feedback.db")

    def test_log_implicit_stores_entry(self):
        entry = make_entry("q1")
        self.db.log_implicit(entry)
        rows = self.db.get_feedback_since(0)
        assert len(rows) == 1
        assert rows[0]["query_id"] == "q1"
        assert rows[0]["cycles_used"] == 1

    def test_log_explicit_rating(self):
        self.db.log_implicit(make_entry("q2"))
        self.db.log_explicit_rating("q2", rating=2)
        rows = self.db.get_feedback_since(0)
        assert rows[0]["user_rating"] == 2

    def test_log_explicit_correction(self):
        self.db.log_implicit(make_entry("q3"))
        self.db.log_explicit_correction("q3", "The correct answer is multi-head attention.")
        rows = self.db.get_feedback_since(0)
        assert "multi-head attention" in rows[0]["user_correction"]

    def test_get_failure_patterns(self):
        # Add mix of success, low_relevance, multi-cycle
        self.db.log_implicit(make_entry("q4", cycles=1, event="success"))
        self.db.log_implicit(make_entry("q5", cycles=2, event="low_relevance", s_rel=0.45))
        self.db.log_implicit(make_entry("q6", cycles=3, event="low_consistency", s_cons=0.55))

        patterns = self.db.get_failure_patterns(since_timestamp=0)

        assert patterns["total_queries"] == 3
        assert patterns["multi_cycle_rate"] > 0
        assert 0.0 <= patterns["low_relevance_rate"] <= 1.0

    def test_total_query_count(self):
        for i in range(5):
            self.db.log_implicit(make_entry(f"q{i}"))
        assert self.db.get_total_query_count() == 5


class TestFeedbackCollector:

    def setup_method(self):
        self.tmpdir   = tempfile.mkdtemp()
        self.collector = FeedbackCollector(db_path=f"{self.tmpdir}/fc_feedback.db")

    def test_record_increments_count(self):
        self.collector.record(make_entry("qa"))
        self.collector.record(make_entry("qb"))
        assert self.collector.total_queries() == 2

    def test_explicit_feedback_roundtrip(self):
        self.collector.record(make_entry("qc"))
        self.collector.user_rates("qc", rating=5)
        self.collector.user_corrects("qc", "The answer should mention positional encodings.")
        rows = self.collector.db.get_feedback_since(0)
        assert rows[0]["user_rating"] == 5
        assert "positional encodings" in rows[0]["user_correction"]
