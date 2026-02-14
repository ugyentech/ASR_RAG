"""
Tests for Layer 1 — Real-Time Self-Repair Module
"""
import pytest
from unittest.mock import MagicMock, patch
from src.layers.layer1_self_repair import SelfRepairModule, FeedbackEntry


def make_mock_retriever(relevance=0.8):
    """Return a retriever that always gives back one document."""
    r = MagicMock()
    r.retrieve.return_value = [
        {"id": "doc1", "text": "The Transformer uses multi-head self-attention.", "source": "paper1", "section": "methods"}
    ]
    return r


def make_mock_llm(answer="The model uses multi-head attention."):
    llm = MagicMock()
    response = MagicMock()
    response.content = answer
    llm.invoke.return_value = response
    return llm


class TestSelfRepairModule:

    def setup_method(self):
        """Set up a SelfRepairModule with mocked dependencies."""
        self.retriever = make_mock_retriever()
        self.llm       = make_mock_llm()

        with patch("src.layers.layer1_self_repair.ConsistencyChecker") as mock_cc, \
             patch("src.layers.layer1_self_repair.ScientificQueryExpander"):

            mock_cc.return_value.score.return_value = 0.85  # above threshold

            self.module = SelfRepairModule(
                retriever          = self.retriever,
                llm                = self.llm,
                relevance_thresh   = 0.60,
                consistency_thresh = 0.70,
                max_iterations     = 3,
            )
            # Inject mocked consistency checker
            self.module.consistency_checker = mock_cc.return_value

    def test_returns_repair_result(self):
        with patch.object(self.module, "_score_relevance", return_value=0.85):
            result = self.module.process("What attention does Transformer use?", "q1")
        assert result.answer
        assert isinstance(result.feedback, FeedbackEntry)
        assert result.success

    def test_single_cycle_on_high_scores(self):
        with patch.object(self.module, "_score_relevance", return_value=0.90):
            result = self.module.process("Test query", "q2")
        assert result.feedback.cycles_used == 1

    def test_reformulates_on_low_relevance(self):
        scores = iter([0.40, 0.40, 0.85])  # low, low, then good
        with patch.object(self.module, "_score_relevance", side_effect=scores):
            result = self.module.process("Ambiguous query", "q3")
        assert result.feedback.cycles_used >= 2
        assert result.feedback.event_type in ("low_relevance", "success")

    def test_max_iterations_respected(self):
        with patch.object(self.module, "_score_relevance", return_value=0.30):
            result = self.module.process("Very ambiguous query", "q4")
        assert result.feedback.cycles_used <= self.module.N

    def test_feedback_entry_has_required_fields(self):
        with patch.object(self.module, "_score_relevance", return_value=0.80):
            result = self.module.process("Query with feedback", "q5")
        fb = result.feedback
        assert fb.query_id == "q5"
        assert fb.latency_ms > 0
        assert isinstance(fb.retrieved_doc_ids, list)
        assert fb.original_query == "Query with feedback"
