"""
Baseline Systems
=================
Four baseline RAG systems evaluated against ASR-RAG:

  1. VanillaRAG   — single-pass BM25 + generation, no correction
  2. SelfRAG      — reflection tokens (Asai et al., 2023)
  3. FLARE        — forward-looking active retrieval (Jiang et al., 2023)
  4. MemPrompt    — growing correction memory (Madaan et al., 2022)

All baselines share the same LLM, embedding model, and document index
as ASR-RAG to ensure fair comparison.
"""

from __future__ import annotations
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 1. Vanilla RAG
# ══════════════════════════════════════════════════════════════════════

class VanillaRAG:
    """
    Standard single-pass RAG — no self-repair, no feedback.
    Baseline that directly measures the benefit of ASR-RAG's layers.
    """

    def __init__(self, retriever, llm, top_k: int = 5):
        self.retriever = retriever
        self.llm       = llm
        self.top_k     = top_k

        self._prompt = (
            "Answer the question using only the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:"
        )

    def query(self, question: str) -> dict:
        docs    = self.retriever.retrieve(question)
        context = "\n\n".join(d["text"] for d in docs)
        prompt  = self._prompt.format(context=context, question=question)
        result  = self.llm.invoke(prompt)
        answer  = result.content if hasattr(result, "content") else str(result)

        return {
            "answer":      answer,
            "context":     docs,
            "cycles_used": 1,
        }


# ══════════════════════════════════════════════════════════════════════
# 2. Self-RAG
# ══════════════════════════════════════════════════════════════════════

class SelfRAG:
    """
    Simplified Self-RAG with four reflection decisions:
      [Retrieve]  — should we retrieve at all?
      [IsRel]     — is the passage relevant?
      [IsSup]     — does the passage support the answer?
      [IsUse]     — is the answer useful?

    Reference: Asai et al. (2023). Self-RAG: Learning to retrieve,
    generate, and critique through self-reflection.
    """

    def __init__(self, retriever, llm, top_k: int = 5):
        self.retriever = retriever
        self.llm       = llm
        self.top_k     = top_k

    def query(self, question: str) -> dict:
        # Step 1: decide whether to retrieve
        if not self._should_retrieve(question):
            answer = self._generate_no_retrieval(question)
            return {"answer": answer, "context": [], "cycles_used": 0}

        # Step 2: retrieve + filter by relevance
        docs = self.retriever.retrieve(question)
        relevant_docs = [d for d in docs if self._is_relevant(question, d["text"])]
        if not relevant_docs:
            relevant_docs = docs[:2]   # fallback to top-2

        # Step 3: generate + check support
        context = "\n\n".join(d["text"] for d in relevant_docs)
        answer  = self._generate(question, context)

        supported = self._is_supported(answer, context)
        if not supported:
            # Regenerate with stricter grounding instruction
            answer = self._generate_grounded(question, context)

        is_useful = self._is_useful(question, answer)
        logger.debug(f"[SelfRAG] supported={supported}, useful={is_useful}")

        return {
            "answer":      answer,
            "context":     relevant_docs,
            "cycles_used": 1,
            "supported":   supported,
            "useful":      is_useful,
        }

    def _should_retrieve(self, question: str) -> bool:
        prompt = f"Should I retrieve external documents to answer: '{question}'? Reply YES or NO."
        result = self.llm.invoke(prompt)
        text   = result.content if hasattr(result, "content") else str(result)
        return "yes" in text.lower()

    def _is_relevant(self, question: str, passage: str) -> bool:
        prompt = (f"Is this passage relevant to the question?\n"
                  f"Question: {question}\nPassage: {passage[:300]}\nReply YES or NO.")
        result = self.llm.invoke(prompt)
        text   = result.content if hasattr(result, "content") else str(result)
        return "yes" in text.lower()

    def _is_supported(self, answer: str, context: str) -> bool:
        prompt = (f"Is this answer fully supported by the context?\n"
                  f"Context: {context[:500]}\nAnswer: {answer}\nReply YES or NO.")
        result = self.llm.invoke(prompt)
        text   = result.content if hasattr(result, "content") else str(result)
        return "yes" in text.lower()

    def _is_useful(self, question: str, answer: str) -> bool:
        prompt = (f"Is this a useful answer to the question?\n"
                  f"Question: {question}\nAnswer: {answer}\nReply YES or NO.")
        result = self.llm.invoke(prompt)
        text   = result.content if hasattr(result, "content") else str(result)
        return "yes" in text.lower()

    def _generate(self, question: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        result = self.llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)

    def _generate_grounded(self, question: str, context: str) -> str:
        prompt = (f"Answer using ONLY the exact information in the context.\n"
                  f"Context:\n{context}\n\nQuestion: {question}\n\nGrounded Answer:")
        result = self.llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)

    def _generate_no_retrieval(self, question: str) -> str:
        result = self.llm.invoke(f"Answer this question: {question}")
        return result.content if hasattr(result, "content") else str(result)


# ══════════════════════════════════════════════════════════════════════
# 3. FLARE
# ══════════════════════════════════════════════════════════════════════

class FLARE:
    """
    Forward-Looking Active REtrieval.
    Triggers retrieval when the model generates a low-confidence token.

    Reference: Jiang et al. (2023). Active retrieval augmented generation.
    """

    def __init__(
        self,
        retriever,
        llm,
        confidence_threshold: float = 0.50,
        top_k: int = 5,
        max_retrieval_steps: int = 3,
    ):
        self.retriever   = retriever
        self.llm         = llm
        self.threshold   = confidence_threshold
        self.top_k       = top_k
        self.max_steps   = max_retrieval_steps

    def query(self, question: str) -> dict:
        """Generate iteratively, triggering retrieval on low-confidence spans."""
        all_context = []
        answer_parts = []
        step = 0

        current_question = question

        while step < self.max_steps:
            # Generate a partial answer and estimate confidence
            partial, confidence = self._generate_with_confidence(
                current_question, all_context
            )

            if confidence >= self.threshold or step == self.max_steps - 1:
                answer_parts.append(partial)
                break

            # Low confidence → retrieve more context
            logger.debug(f"[FLARE] Step {step+1}: confidence={confidence:.2f} < "
                         f"{self.threshold}. Retrieving.")
            new_docs = self.retriever.retrieve(current_question)
            all_context.extend(new_docs)
            # De-duplicate by id
            seen = set()
            all_context = [
                d for d in all_context
                if d["id"] not in seen and not seen.add(d["id"])
            ]
            answer_parts.append(partial)
            step += 1

        final_answer = " ".join(answer_parts).strip()
        if not final_answer:
            # Fallback: generate once with all accumulated context
            context_text = "\n\n".join(d["text"] for d in all_context)
            prompt  = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            result  = self.llm.invoke(prompt)
            final_answer = result.content if hasattr(result, "content") else str(result)

        return {
            "answer":      final_answer,
            "context":     all_context,
            "cycles_used": step + 1,
        }

    def _generate_with_confidence(
        self, question: str, context_docs: list[dict]
    ) -> tuple[str, float]:
        """
        Generate a partial answer and return (text, confidence_proxy).
        In production use actual token probabilities from the LLM logits.
        Here we use a heuristic: responses with hedging words → low confidence.
        """
        context_text = "\n\n".join(d["text"] for d in context_docs) if context_docs else ""
        prompt = (
            f"{'Context:' + chr(10) + context_text + chr(10) + chr(10) if context_text else ''}"
            f"Question: {question}\n\nPartial Answer:"
        )
        result = self.llm.invoke(prompt)
        text   = result.content if hasattr(result, "content") else str(result)

        # Heuristic confidence proxy
        hedge_words = ["i think", "i believe", "perhaps", "might", "unclear",
                       "not sure", "i don't know", "it's possible"]
        confidence = 0.3 if any(h in text.lower() for h in hedge_words) else 0.75
        return text, confidence


# ══════════════════════════════════════════════════════════════════════
# 4. MemPrompt
# ══════════════════════════════════════════════════════════════════════

class MemPrompt:
    """
    Memory-assisted prompt editing with user feedback.
    Maintains a growing list of past corrections; retrieves the
    most similar ones to prepend to each new query's prompt.

    Reference: Madaan et al. (2022). MemPrompt: Memory-assisted
    prompt editing with user feedback.
    """

    def __init__(
        self,
        retriever,
        llm,
        top_k: int = 5,
        memory_top_k: int = 3,
    ):
        self.retriever    = retriever
        self.llm          = llm
        self.top_k        = top_k
        self.memory_top_k = memory_top_k
        self.memory: list[dict] = []   # {question, correction, embedding}

    def query(self, question: str) -> dict:
        # 1. Find relevant past corrections
        past_corrections = self._retrieve_corrections(question)

        # 2. Build augmented prompt
        prompt = self._build_prompt(question, past_corrections)

        # 3. Retrieve passages
        docs    = self.retriever.retrieve(question)
        context = "\n\n".join(d["text"] for d in docs)

        # 4. Generate with memory-augmented prompt
        full_prompt = f"{prompt}\n\nContext:\n{context}\n\nAnswer:"
        result      = self.llm.invoke(full_prompt)
        answer      = result.content if hasattr(result, "content") else str(result)

        return {
            "answer":            answer,
            "context":           docs,
            "cycles_used":       1,
            "corrections_used":  len(past_corrections),
        }

    def add_correction(self, question: str, correction: str):
        """
        User provides a correction — stored in memory for future queries.
        Called from evaluation harness when ground truth differs from answer.
        """
        self.memory.append({
            "question":   question,
            "correction": correction,
        })
        logger.debug(f"[MemPrompt] Memory size: {len(self.memory)}")

    def _retrieve_corrections(self, question: str) -> list[dict]:
        """BM25 over past questions to find similar corrections."""
        if not self.memory:
            return []

        from rank_bm25 import BM25Okapi
        corpus    = [m["question"].lower().split() for m in self.memory]
        bm25      = BM25Okapi(corpus)
        scores    = bm25.get_scores(question.lower().split())
        top_idxs  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_idxs  = top_idxs[:self.memory_top_k]
        return [self.memory[i] for i in top_idxs if scores[i] > 0]

    @staticmethod
    def _build_prompt(question: str, corrections: list[dict]) -> str:
        if not corrections:
            return f"Answer the following question accurately:\nQuestion: {question}"

        correction_text = "\n".join(
            f"- For '{c['question']}': {c['correction']}"
            for c in corrections
        )
        return (
            f"Here are corrections from past similar questions:\n"
            f"{correction_text}\n\n"
            f"Use these insights to answer accurately:\n"
            f"Question: {question}"
        )
