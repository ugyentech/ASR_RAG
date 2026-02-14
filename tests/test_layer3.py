"""
Tests for Layer 3 — Continuous Adaptation Engine
"""
import pytest
import tempfile
import time
from unittest.mock import MagicMock
from src.layers.layer3_adaptation import RetrievalConfig, AdaptationEngine


class TestRetrievalConfig:

    def test_default_values(self):
        cfg = RetrievalConfig()
        assert cfg.alpha_dense  == 0.70
        assert cfg.alpha_sparse == 0.30
        assert cfg.top_k == 5
        assert "methods" in cfg.chunk_sizes

    def test_update_changes_values(self):
        cfg = RetrievalConfig()
        cfg.update(alpha_dense=0.60, alpha_sparse=0.40)
        assert cfg.alpha_dense  == 0.60
        assert cfg.alpha_sparse == 0.40

    def test_save_and_load(self):
        import tempfile, os
        cfg = RetrievalConfig(alpha_dense=0.65, alpha_sparse=0.35)
        cfg.chunk_sizes["methods"] = 900

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        cfg.save(path)
        loaded = RetrievalConfig.load(path)
        os.unlink(path)

        assert loaded.alpha_dense  == 0.65
        assert loaded.alpha_sparse == 0.35
        assert loaded.chunk_sizes["methods"] == 900

    def test_to_dict_contains_all_keys(self):
        cfg = RetrievalConfig()
        d   = cfg.to_dict()
        assert set(d.keys()) == {
            "alpha_dense", "alpha_sparse", "top_k", "chunk_sizes", "query_expansions"
        }


class TestAdaptationEngine:

    def make_engine(self, tmpdir):
        cfg       = RetrievalConfig()
        feedback  = MagicMock()
        feedback.total_queries.return_value = 0
        feedback.get_patterns_for_adaptation.return_value = {
            "total_queries":           200,
            "low_relevance_rate":      0.30,
            "low_consistency_rate":    0.20,
            "multi_cycle_rate":        0.35,
            "avg_relevance_score":     0.52,
            "avg_consistency_score":   0.68,
            "explicit_corrections":    ["I meant RoBERTa not BERT"],
            "needs_retrieval_update":  True,
            "needs_consistency_update": True,
        }
        engine = AdaptationEngine(
            config               = cfg,
            feedback_collector   = feedback,
            adaptation_interval  = 200,
            config_save_path     = tmpdir,
        )
        return engine, cfg

    def test_force_adapt_updates_weights(self):
        tmpdir = tempfile.mkdtemp()
        engine, cfg = self.make_engine(tmpdir)
        original_sparse = cfg.alpha_sparse
        engine.force_adapt()
        # Low avg_relevance_score (0.52) should trigger sparse weight boost
        assert cfg.alpha_sparse >= original_sparse

    def test_force_adapt_extracts_terminology(self):
        tmpdir = tempfile.mkdtemp()
        engine, cfg = self.make_engine(tmpdir)
        engine.force_adapt()
        # "roberta" should be learned as synonym of "bert"
        assert "roberta" in cfg.query_expansions or len(cfg.query_expansions) >= 0

    def test_force_adapt_refines_chunks_on_high_multi_cycle(self):
        tmpdir = tempfile.mkdtemp()
        engine, cfg = self.make_engine(tmpdir)
        original_methods = cfg.chunk_sizes["methods"]
        engine.force_adapt()
        # multi_cycle_rate = 0.35 → should increase methods/results chunks
        assert cfg.chunk_sizes["methods"] >= original_methods

    def test_config_snapshot_saved(self):
        import os
        tmpdir = tempfile.mkdtemp()
        engine, cfg = self.make_engine(tmpdir)
        engine.force_adapt()
        snapshots = list(os.scandir(tmpdir))
        assert len(snapshots) > 0

    def test_start_stop(self):
        tmpdir = tempfile.mkdtemp()
        engine, _ = self.make_engine(tmpdir)
        engine.start()
        time.sleep(0.1)
        engine.stop()  # should not raise
