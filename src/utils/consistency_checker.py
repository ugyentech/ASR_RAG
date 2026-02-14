"""
Consistency Checker
====================
Scores how well a generated answer is grounded in retrieved context.
Returns a float in [0, 1] — the faithfulness score.

Two modes:
  - 'nli'   : Natural Language Inference cross-encoder (accurate, slower)
  - 'simple': Token-overlap heuristic (fast, good enough for self-repair loop)
"""

from __future__ import annotations
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """
    Scores faithfulness of an answer against retrieved context.

    Parameters
    ----------
    llm    : LangChain BaseLLM (used for NLI-style prompting in 'llm' mode)
    mode   : 'llm' | 'simple'
      - 'llm'    : prompts the LLM to verify each claim (most accurate)
      - 'simple' : token-overlap heuristic (fastest, no extra LLM call)
    """

    def __init__(self, llm=None, mode: Literal["llm", "simple"] = "llm"):
        self.llm  = llm
        self.mode = mode

        self._verify_prompt = (
            "You are a fact-checker. For each claim in the answer, "
            "state whether it is SUPPORTED or NOT SUPPORTED by the context.\n\n"
            "Context:\n{context}\n\n"
            "Answer to check:\n{answer}\n\n"
            "List each claim and its verdict (SUPPORTED / NOT SUPPORTED).\n"
            "At the end write: SCORE: X/Y where X=supported claims, Y=total claims."
        )

    def score(self, answer: str, context_docs: list[dict]) -> float:
        """
        Return faithfulness score in [0, 1].
        1.0 = all claims verifiable in context.
        0.0 = no claims verifiable.
        """
        if not answer or not context_docs:
            return 0.0

        if self.mode == "llm" and self.llm is not None:
            return self._llm_score(answer, context_docs)
        return self._simple_score(answer, context_docs)

    # ── LLM-based scoring ──────────────────────────────────────────────

    def _llm_score(self, answer: str, context_docs: list[dict]) -> float:
        context_text = "\n\n".join(d["text"] for d in context_docs)
        prompt = self._verify_prompt.format(
            context=context_text[:3000],   # truncate to avoid context overflow
            answer=answer,
        )
        try:
            result = self.llm.invoke(prompt)
            text   = result.content if hasattr(result, "content") else str(result)
            return self._parse_score(text)
        except Exception as e:
            logger.warning(f"[ConsistencyChecker] LLM scoring failed: {e}. Falling back.")
            return self._simple_score(answer, context_docs)

    @staticmethod
    def _parse_score(llm_output: str) -> float:
        """Parse 'SCORE: X/Y' from LLM output."""
        match = re.search(r"SCORE:\s*(\d+)\s*/\s*(\d+)", llm_output, re.IGNORECASE)
        if match:
            supported, total = int(match.group(1)), int(match.group(2))
            return supported / total if total > 0 else 0.0
        # Fallback: count SUPPORTED vs NOT SUPPORTED mentions
        supported = llm_output.upper().count("SUPPORTED") - llm_output.upper().count("NOT SUPPORTED")
        total     = llm_output.upper().count("SUPPORTED")
        return max(0.0, supported / total) if total > 0 else 0.5

    # ── Simple token-overlap scoring ───────────────────────────────────

    @staticmethod
    def _simple_score(answer: str, context_docs: list[dict]) -> float:
        """
        Rough faithfulness proxy:
        fraction of answer bigrams present in context.
        Fast enough to use inside the repair loop.
        """
        def bigrams(text: str) -> set:
            tokens = re.findall(r"\b\w+\b", text.lower())
            return {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}

        context_text = " ".join(d["text"] for d in context_docs)
        ans_bigrams  = bigrams(answer)
        ctx_bigrams  = bigrams(context_text)

        if not ans_bigrams:
            return 1.0   # empty answer — nothing to hallucinate
        overlap = ans_bigrams & ctx_bigrams
        return len(overlap) / len(ans_bigrams)
