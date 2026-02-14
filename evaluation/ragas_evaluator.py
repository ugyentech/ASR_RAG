"""
RAGAS Evaluator
================
Computes Answer Relevancy, Faithfulness, Context Precision,
and Context Recall for a batch of query results.

Compatible with ragas>=0.1.0
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """
    Wraps the ragas library to evaluate ASR-RAG outputs.

    Parameters
    ----------
    results_dir : str — where to save evaluation CSVs
    """

    def __init__(self, results_dir: str = "results/"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        samples: list[dict],
        run_name: str = "eval",
        save: bool = True,
    ) -> dict:
        """
        Evaluate a list of QA samples.

        Each sample must have:
          - question   : str
          - answer     : str   (generated answer)
          - contexts   : list[str]  (retrieved passage texts)
          - ground_truth: str  (reference answer, optional for some metrics)

        Returns dict of metric → score.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            df      = pd.DataFrame(samples)
            dataset = Dataset.from_pandas(df)

            result = evaluate(
                dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    context_precision,
                    context_recall,
                ],
            )

            scores = {
                "answer_relevancy":  result["answer_relevancy"],
                "faithfulness":      result["faithfulness"],
                "context_precision": result["context_precision"],
                "context_recall":    result["context_recall"],
            }

        except ImportError:
            logger.warning("ragas not installed. Computing approximate scores.")
            scores = self._approximate_scores(samples)

        logger.info(f"[Evaluator] {run_name}: " +
                    " | ".join(f"{k}={v:.3f}" for k, v in scores.items()))

        if save:
            self._save(scores, samples, run_name)

        return scores

    def evaluate_over_time(
        self,
        pipeline,
        queries: list[dict],
        record_every: int = 200,
        run_name: str = "temporal",
    ) -> list[dict]:
        """
        Run all queries through the pipeline, recording metrics
        every `record_every` queries to produce a learning curve.

        queries : list of {"question": str, "ground_truth": str}
        Returns : list of {cycle, queries_processed, ...metrics}
        """
        timeline = []
        samples  = []

        for i, q in enumerate(queries, 1):
            result = pipeline.query(q["question"])
            samples.append({
                "question":    q["question"],
                "answer":      result["answer"],
                "contexts":    [d["text"] for d in result["context"]],
                "ground_truth": q.get("ground_truth", ""),
            })

            if i % record_every == 0 or i == len(queries):
                batch_samples = samples[-record_every:]
                scores = self.evaluate(
                    batch_samples,
                    run_name=f"{run_name}_cycle{i // record_every}",
                    save=True,
                )
                timeline.append({
                    "cycle":             i // record_every,
                    "queries_processed": i,
                    **scores,
                })
                logger.info(f"[Evaluator] Cycle {i // record_every} complete "
                            f"({i}/{len(queries)} queries)")

        return timeline

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _approximate_scores(samples: list[dict]) -> dict:
        """
        Lightweight approximate metrics when ragas is not available.
        Uses simple token overlap — not suitable for final reporting.
        """
        import re

        def tokens(text: str) -> set:
            return set(re.findall(r"\b\w+\b", text.lower()))

        relevancy_scores = []
        faithfulness_scores = []

        for s in samples:
            q_toks = tokens(s["question"])
            a_toks = tokens(s["answer"])
            c_toks = tokens(" ".join(s.get("contexts", [])))

            # Rough answer relevancy: question-answer token overlap
            if q_toks:
                relevancy_scores.append(len(q_toks & a_toks) / len(q_toks))

            # Rough faithfulness: answer tokens in context
            if a_toks:
                faithfulness_scores.append(len(a_toks & c_toks) / len(a_toks))

        return {
            "answer_relevancy":  sum(relevancy_scores) / max(len(relevancy_scores), 1),
            "faithfulness":      sum(faithfulness_scores) / max(len(faithfulness_scores), 1),
            "context_precision": 0.0,   # cannot compute without ground truth docs
            "context_recall":    0.0,
        }

    def _save(self, scores: dict, samples: list[dict], run_name: str):
        # Save scores JSON
        score_path = self.results_dir / f"{run_name}_scores.json"
        with open(score_path, "w") as f:
            json.dump(scores, f, indent=2)

        # Save sample-level results CSV
        df = pd.DataFrame(samples)
        df["run_name"] = run_name
        df.to_csv(self.results_dir / f"{run_name}_samples.csv", index=False)

        logger.info(f"[Evaluator] Results saved to {self.results_dir}")
