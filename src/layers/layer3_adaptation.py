"""
Layer 3: Continuous Adaptation Engine
=======================================
Implements Algorithm 2 from the ASR-RAG paper.

Runs asynchronously every Δt queries. Reads failure patterns from
Layer 2's FeedbackDB and updates the retrieval configuration:

  1. Hybrid weight optimisation (α_dense / α_sparse per query type)
  2. Scientific terminology extraction → embedding updates
  3. Section-aware chunk boundary refinement
  4. Reranking calibration
  5. Hot-swap deployment (zero downtime)
"""

from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RetrievalConfig:
    """
    Mutable retrieval configuration that can be hot-swapped at runtime.
    All Layer 1 components read from this object — when Layer 3 updates
    it, the next query automatically uses the new settings.
    """

    def __init__(
        self,
        alpha_dense: float = 0.70,
        alpha_sparse: float = 0.30,
        top_k: int = 5,
        chunk_sizes: Optional[dict] = None,
        query_expansions: Optional[dict] = None,
    ):
        self.alpha_dense      = alpha_dense
        self.alpha_sparse     = alpha_sparse
        self.top_k            = top_k
        self.chunk_sizes      = chunk_sizes or {
            "abstract": 250, "introduction": 512, "methods": 768,
            "results": 768, "conclusion": 300, "default": 512,
        }
        self.query_expansions = query_expansions or {}   # term → [synonyms]
        self._lock            = threading.RLock()

    def update(self, **kwargs):
        """Thread-safe in-place update."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            logger.info(f"[Layer3] Config updated: {kwargs}")

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "alpha_dense":      self.alpha_dense,
                "alpha_sparse":     self.alpha_sparse,
                "top_k":            self.top_k,
                "chunk_sizes":      self.chunk_sizes,
                "query_expansions": self.query_expansions,
            }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RetrievalConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class AdaptationEngine:
    """
    Layer 3 — Continuous Adaptation Engine.

    Parameters
    ----------
    config          : RetrievalConfig  — shared mutable config object
    feedback_collector : FeedbackCollector — Layer 2 interface
    adaptation_interval : int          — Δt queries between cycles
    config_save_path    : str          — where to persist config snapshots
    """

    def __init__(
        self,
        config,
        feedback_collector,
        adaptation_interval: int = 200,
        config_save_path: str = "data/config_snapshots/",
        success_rate_threshold: float = 0.70,
    ):
        self.config              = config
        self.feedback            = feedback_collector
        self.interval            = adaptation_interval
        self.save_path           = Path(config_save_path)
        self.success_threshold   = success_rate_threshold
        self._cycle_count        = 0
        self._last_query_count   = 0
        self._running            = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────

    def start(self):
        """Start the background adaptation thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[Layer3] Adaptation engine started.")

    def stop(self):
        """Gracefully stop the background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("[Layer3] Adaptation engine stopped.")

    def force_adapt(self):
        """Manually trigger one adaptation cycle (useful for testing)."""
        self._run_adaptation_cycle()

    # ── Background loop ────────────────────────────────────────────────

    def _run_loop(self):
        """Algorithm 2 — runs in background thread."""
        while self._running:
            current_count = self.feedback.total_queries()
            queries_since = current_count - self._last_query_count

            if queries_since >= self.interval:
                logger.info(f"[Layer3] Δt={queries_since} queries reached. "
                            f"Starting adaptation cycle {self._cycle_count + 1}.")
                self._run_adaptation_cycle()
                self._last_query_count = current_count

            time.sleep(30)   # check every 30 seconds

    def _run_adaptation_cycle(self):
        """One full adaptation cycle — Algorithm 2, lines 3–15."""
        self._cycle_count += 1
        cycle_id = self._cycle_count
        logger.info(f"[Layer3] === Cycle {cycle_id} begin ===")

        # Line 3: collect feedback since last cycle
        patterns = self.feedback.get_patterns_for_adaptation()
        if not patterns:
            logger.info("[Layer3] No feedback yet. Skipping cycle.")
            return

        logger.info(f"[Layer3] Patterns: "
                    f"low_rel={patterns['low_relevance_rate']:.2f}, "
                    f"low_cons={patterns['low_consistency_rate']:.2f}, "
                    f"multi_cycle={patterns['multi_cycle_rate']:.2f}")

        updates = {}

        # Lines 5-10: optimise hybrid weights per failure pattern
        if patterns["needs_retrieval_update"]:
            updates.update(self._optimise_hybrid_weights(patterns))

        # Lines 11-12: extract terminology → update query expansions
        if patterns["explicit_corrections"]:
            new_expansions = self._extract_terminology(
                patterns["explicit_corrections"],
                self.config.query_expansions,
            )
            updates["query_expansions"] = new_expansions

        # Line 13: refine chunk boundaries
        if patterns["multi_cycle_rate"] > 0.30:
            updates["chunk_sizes"] = self._refine_chunk_boundaries(patterns)

        # Apply all updates atomically (hot-swap)
        if updates:
            self.config.update(**updates)
            snapshot_path = str(
                self.save_path / f"config_cycle_{cycle_id}.json"
            )
            self.config.save(snapshot_path)
            logger.info(f"[Layer3] Config saved to {snapshot_path}")

        logger.info(f"[Layer3] === Cycle {cycle_id} complete ===")

    # ── Adaptation strategies ──────────────────────────────────────────

    def _optimise_hybrid_weights(self, patterns: dict) -> dict:
        """
        Shift weight toward sparse BM25 when retrieval precision is low
        (notation-heavy queries benefit from exact keyword matching).
        Shift toward dense when recall is low (conceptual queries).
        """
        low_rel  = patterns["low_relevance_rate"]
        avg_rel  = patterns["avg_relevance_score"]

        current_dense  = self.config.alpha_dense
        current_sparse = self.config.alpha_sparse

        if avg_rel < 0.55 and low_rel > 0.25:
            # Boost sparse — likely notation / exact-match failures
            new_sparse = min(current_sparse + 0.05, 0.50)
            new_dense  = round(1.0 - new_sparse, 2)
        elif avg_rel < 0.65 and low_rel > 0.15:
            # Mild boost to sparse
            new_sparse = min(current_sparse + 0.03, 0.45)
            new_dense  = round(1.0 - new_sparse, 2)
        else:
            return {}  # no change needed

        logger.info(f"[Layer3] Hybrid weights: "
                    f"dense {current_dense:.2f}→{new_dense:.2f}, "
                    f"sparse {current_sparse:.2f}→{new_sparse:.2f}")
        return {"alpha_dense": new_dense, "alpha_sparse": new_sparse}

    def _extract_terminology(
        self,
        corrections: list[str],
        existing_expansions: dict,
    ) -> dict:
        """
        Extract new domain terms from user corrections and build
        a synonym map for query expansion.

        In production replace the simple heuristic below with a
        named-entity recogniser fine-tuned on scientific text.
        """
        import re
        updated = dict(existing_expansions)

        for correction in corrections:
            # Look for patterns like "I meant X not Y" or "X is also called Y"
            patterns = [
                r"I meant ([A-Za-z0-9\-]+) not ([A-Za-z0-9\-]+)",
                r"([A-Za-z0-9\-]+) is also (?:called|known as) ([A-Za-z0-9\-]+)",
                r"use ([A-Za-z0-9\-]+) instead of ([A-Za-z0-9\-]+)",
            ]
            for pat in patterns:
                matches = re.findall(pat, correction, re.IGNORECASE)
                for correct_term, wrong_term in matches:
                    key = correct_term.lower()
                    if key not in updated:
                        updated[key] = []
                    if wrong_term.lower() not in updated[key]:
                        updated[key].append(wrong_term.lower())
                        logger.info(f"[Layer3] New synonym: '{key}' ↔ '{wrong_term}'")

        return updated

    def _refine_chunk_boundaries(self, patterns: dict) -> dict:
        """
        Adjust section chunk sizes based on multi-cycle failure rate.
        If many failures occur on method/results queries → increase
        those section sizes to capture more context per chunk.
        """
        current = dict(self.config.chunk_sizes)
        multi_cycle_rate = patterns["multi_cycle_rate"]

        if multi_cycle_rate > 0.40:
            # Significant context fragmentation — increase all sizes by 10%
            adjusted = {
                k: min(int(v * 1.10), 1024)
                for k, v in current.items()
            }
            logger.info("[Layer3] Chunk sizes increased by 10% due to high multi-cycle rate.")
            return adjusted
        elif multi_cycle_rate > 0.30:
            # Moderate — only increase dense sections
            adjusted = dict(current)
            for section in ["methods", "results"]:
                adjusted[section] = min(int(adjusted[section] * 1.08), 1024)
            logger.info("[Layer3] Methods/Results chunk sizes increased by 8%.")
            return adjusted

        return current  # no change
