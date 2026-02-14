"""
Scientific Query Expander
==========================
Expands user queries with domain synonyms, acronym resolution,
and related concept injection — learned from Layer 2 feedback.
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)

# Static scientific acronym map (seed vocabulary)
ACRONYM_MAP: dict[str, str] = {
    "llm":   "large language model",
    "nlp":   "natural language processing",
    "ml":    "machine learning",
    "dl":    "deep learning",
    "rl":    "reinforcement learning",
    "cv":    "computer vision",
    "rag":   "retrieval augmented generation",
    "bert":  "bidirectional encoder representations transformers",
    "gpt":   "generative pre-trained transformer",
    "transformer": "attention mechanism encoder decoder",
    "vae":   "variational autoencoder",
    "gan":   "generative adversarial network",
    "cnn":   "convolutional neural network",
    "rnn":   "recurrent neural network",
    "lstm":  "long short-term memory",
    "sgd":   "stochastic gradient descent",
    "adam":  "adaptive moment estimation optimizer",
    "bleu":  "bilingual evaluation understudy score",
    "rouge": "recall-oriented understudy gisting evaluation",
    "f1":    "f1 score precision recall harmonic mean",
    "sft":   "supervised fine tuning",
    "rlhf":  "reinforcement learning from human feedback",
    "peft":  "parameter efficient fine tuning",
    "lora":  "low rank adaptation",
}


class ScientificQueryExpander:
    """
    Expands a query that failed relevance scoring with:
      1. Acronym resolution from a static + learned map
      2. Synonym injection from Layer 3's learned vocabulary
      3. Section-hint appending (e.g., adding 'methods' for how-to queries)
    """

    def __init__(self, learned_expansions: dict | None = None):
        # learned_expansions comes from RetrievalConfig.query_expansions
        self.learned: dict[str, list[str]] = learned_expansions or {}

    def update_learned(self, expansions: dict[str, list[str]]):
        """Called by Layer 3 when config is hot-swapped."""
        self.learned = expansions
        logger.info(f"[QueryExpander] Updated with {len(expansions)} learned terms.")

    def expand(self, query: str, context_docs: list[dict] | None = None) -> str:
        """
        Return an expanded version of the query.

        Strategy:
          1. Resolve acronyms → full form
          2. Inject learned synonyms
          3. Append section hint based on query intent
        """
        expanded = query

        # ── 1. Acronym resolution ───────────────────────────────────
        tokens = query.lower().split()
        expansions_added = []
        for token in tokens:
            clean = re.sub(r"[^a-z0-9]", "", token)
            if clean in ACRONYM_MAP:
                expansions_added.append(ACRONYM_MAP[clean])
            elif clean in self.learned:
                expansions_added.extend(self.learned[clean])

        if expansions_added:
            expanded = f"{query} ({' '.join(set(expansions_added))})"

        # ── 2. Section hint ─────────────────────────────────────────
        hint = self._detect_section_hint(query)
        if hint:
            expanded = f"{expanded} [{hint}]"

        logger.debug(f"[QueryExpander] '{query}' → '{expanded}'")
        return expanded

    @staticmethod
    def _detect_section_hint(query: str) -> str:
        """Guess which paper section is most likely to contain the answer."""
        q = query.lower()
        if any(w in q for w in ["how", "method", "approach", "procedure", "algorithm", "train"]):
            return "methods section"
        if any(w in q for w in ["result", "performance", "accuracy", "score", "beat", "achieve"]):
            return "results section"
        if any(w in q for w in ["why", "motivation", "propose", "contribution", "novel"]):
            return "introduction section"
        if any(w in q for w in ["compare", "versus", "vs", "baseline", "prior work"]):
            return "related work section"
        if any(w in q for w in ["conclusion", "future", "limitation", "drawback"]):
            return "conclusion section"
        return ""
