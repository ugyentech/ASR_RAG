"""
Layer 1: Real-Time Self-Repair Module
======================================
Implements Algorithm 1 from the ASR-RAG paper.

For each query:
  1. Hybrid retrieval (dense + BM25)
  2. Relevance scoring  → query reformulation if s_rel < τ_r
  3. Response generation
  4. Consistency check  → constrained regeneration if s_cons < τ_c
  5. Log all feedback metadata for Layer 2

Repeats up to N times until thresholds are met.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate

from src.utils.hybrid_retriever import HybridRetriever
from src.utils.consistency_checker import ConsistencyChecker
from src.utils.query_expander import ScientificQueryExpander

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """One feedback record written to Layer 2's database."""
    query_id: str
    original_query: str
    final_query: str
    retrieved_doc_ids: list[str]
    relevance_score: float
    consistency_score: float
    cycles_used: int
    event_type: str          # 'low_relevance' | 'low_consistency' | 'success'
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RepairResult:
    """What Layer 1 returns after processing a query."""
    answer: str
    context: list[dict]
    feedback: FeedbackEntry
    success: bool


class SelfRepairModule:
    """
    Layer 1 — Real-Time Self-Repair Module.

    Parameters
    ----------
    retriever       : HybridRetriever — dense + BM25 hybrid search
    llm             : BaseLLM         — Llama-3-8B or any LangChain LLM
    relevance_thresh: float           — τ_r (default 0.60)
    consistency_thresh: float         — τ_c (default 0.70)
    max_iterations  : int             — N   (default 3)
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: BaseLLM,
        relevance_thresh: float = 0.60,
        consistency_thresh: float = 0.70,
        max_iterations: int = 3,
    ):
        self.retriever          = retriever
        self.llm                = llm
        self.tau_r              = relevance_thresh
        self.tau_c              = consistency_thresh
        self.N                  = max_iterations
        self.query_expander     = ScientificQueryExpander()
        self.consistency_checker = ConsistencyChecker(llm)
        # ADD THIS LINE:
        self._relevance_model    = None   # lazy-loaded and cached

        self._gen_prompt = PromptTemplate.from_template(
            "Use the following context to answer the question.\n"
            "Be factual and only use information from the context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self._grounded_prompt = PromptTemplate.from_template(
            "Answer ONLY using the exact facts stated in the context below.\n"
            "Do not infer or add information not explicitly in the context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Grounded Answer:"
        )

    # ── Public API ─────────────────────────────────────────────────────

    def process(self, query: str, query_id: str = "") -> RepairResult:
        """
        Run Algorithm 1: retrieve → score → reformulate → generate → check.

        Parameters
        ----------
        query    : user's natural language question
        query_id : optional identifier for logging

        Returns
        -------
        RepairResult with answer, context, feedback metadata, and success flag
        """
        start_time   = time.time()
        q_current    = query
        i            = 0
        s_rel        = 0.0
        s_cons       = 0.0
        context_docs = []
        answer       = ""
        event_type   = "success"

        logger.info(f"[Layer1] Processing query_id={query_id}: '{query[:80]}...'")

        while i < self.N:
            # ── Step 1: Hybrid Retrieval ────────────────────────────
            context_docs = self.retriever.retrieve(q_current)
            context_text = self._format_context(context_docs)

            # ── Step 2: Relevance Scoring ───────────────────────────
            s_rel = self._score_relevance(q_current, context_docs)
            logger.debug(f"  Cycle {i+1}: s_rel={s_rel:.3f}")

            if s_rel < self.tau_r:
                logger.info(f"  Low relevance ({s_rel:.3f} < {self.tau_r}). Reformulating.")
                q_current  = self.query_expander.expand(q_current, context_docs)
                event_type = "low_relevance"
                i += 1
                continue

            # ── Step 3: Generate Response ───────────────────────────
            answer = self._generate(q_current, context_text, grounded=False)

            # ── Step 4: Consistency Check ───────────────────────────
            s_cons = self.consistency_checker.score(answer, context_docs)
            logger.debug(f"  Cycle {i+1}: s_cons={s_cons:.3f}")

            if s_cons < self.tau_c:
                logger.info(f"  Low consistency ({s_cons:.3f} < {self.tau_c}). Regenerating.")
                answer     = self._generate(q_current, context_text, grounded=True)
                s_cons     = self.consistency_checker.score(answer, context_docs)
                event_type = "low_consistency"

            i += 1

            # Exit if both thresholds satisfied
            if s_rel >= self.tau_r and s_cons >= self.tau_c:
                event_type = "success"
                break

        latency = (time.time() - start_time) * 1000
        logger.info(f"[Layer1] Done in {i} cycle(s), {latency:.0f} ms. "
                    f"s_rel={s_rel:.3f}, s_cons={s_cons:.3f}")

        feedback = FeedbackEntry(
            query_id        = query_id or str(time.time()),
            original_query  = query,
            final_query     = q_current,
            retrieved_doc_ids = [d["id"] for d in context_docs],
            relevance_score = s_rel,
            consistency_score = s_cons,
            cycles_used     = i,
            event_type      = event_type,
            latency_ms      = latency,
        )

        return RepairResult(
            answer  = answer,
            context = context_docs,
            feedback = feedback,
            success = (s_rel >= self.tau_r and s_cons >= self.tau_c),
        )

    # ── Private helpers ────────────────────────────────────────────────

    def _format_context(self, docs: list[dict]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] (Source: {doc.get('source','?')}, "
                         f"Section: {doc.get('section','?')})\n{doc['text']}")
        return "\n\n".join(parts)

    def _score_relevance(self, query: str, docs: list[dict]) -> float:
        """
        Simple cosine similarity between query embedding and
        mean passage embedding. Replace with a cross-encoder for production.
        """
        if not docs:
            return 0.0
        from sentence_transformers import SentenceTransformer, util

        # Load model once and cache it — saves ~2 seconds per query on laptops
        if self._relevance_model is None:
            self._relevance_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

        q_emb  = self._relevance_model.encode(query, convert_to_tensor=True)
        d_embs = self._relevance_model.encode(
            [d["text"] for d in docs], convert_to_tensor=True
        )
        scores = util.cos_sim(q_emb, d_embs)[0]
        return float(scores.mean())

    def _generate(self, query: str, context: str, grounded: bool) -> str:
        """Call the LLM with the appropriate prompt template."""
        prompt = self._grounded_prompt if grounded else self._gen_prompt
        chain  = prompt | self.llm
        result = chain.invoke({"context": context, "question": query})
        return result.content if hasattr(result, "content") else str(result)
