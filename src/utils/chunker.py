"""
Section-Aware Document Chunker
================================
Splits scientific papers into chunks that respect section boundaries.
Section sizes are controlled by RetrievalConfig and updated by Layer 3.

Supported input formats:
  - Plain text with section headers (## Introduction, # Methods, etc.)
  - Dict with keys: {"title", "abstract", "sections": [{name, text}, ...]}
"""

from __future__ import annotations
import re
import uuid
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Known section name patterns → canonical section key
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"abstract",            "abstract"),
    (r"introduction",        "introduction"),
    (r"related.?work|background|prior.?work", "related_work"),
    (r"method|approach|model|framework|system", "methods"),
    (r"experiment|evaluation|result|finding",   "results"),
    (r"discussion|analysis|ablation",           "discussion"),
    (r"conclusion|future.?work|limitation",     "conclusion"),
    (r"appendix|supplement",                    "appendix"),
]


def detect_section(header: str) -> str:
    """Map a raw section header string to a canonical key."""
    h = header.lower().strip()
    for pattern, key in SECTION_PATTERNS:
        if re.search(pattern, h):
            return key
    return "default"


class SectionAwareChunker:
    """
    Chunks scientific paper text into variable-size segments
    based on which paper section they belong to.

    Parameters
    ----------
    config : RetrievalConfig — live config with chunk_sizes dict
    overlap: int             — token overlap between consecutive chunks
    """

    def __init__(self, config=None, overlap: int = 64):
        self.config  = config
        self.overlap = overlap

    @property
    def _chunk_sizes(self) -> dict:
        if self.config:
            return self.config.chunk_sizes
        # Fallback defaults matching config.yaml
        return {
            "abstract": 250, "introduction": 512, "related_work": 512,
            "methods": 768, "results": 768, "discussion": 512,
            "conclusion": 300, "appendix": 512, "default": 512,
        }

    # ── Public API ─────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        source: str = "",
        mode: Literal["section_aware", "fixed"] = "section_aware",
        fixed_size: int = 512,
    ) -> list[dict]:
        """
        Chunk a raw paper text string.

        Returns list of dicts:
          {id, text, source, section, token_count}
        """
        if mode == "fixed":
            return self._fixed_chunk(text, source, fixed_size)

        sections = self._split_into_sections(text)
        chunks   = []
        for section_name, section_text in sections:
            section_key  = detect_section(section_name)
            target_size  = self._chunk_sizes.get(section_key, self._chunk_sizes["default"])
            section_chunks = self._chunk_section(section_text, source, section_key, target_size)
            chunks.extend(section_chunks)

        logger.debug(f"[Chunker] '{source}' → {len(chunks)} chunks (section-aware)")
        return chunks

    def chunk_dict(self, paper_dict: dict) -> list[dict]:
        """
        Chunk a structured paper dict:
        {"title": "...", "abstract": "...", "sections": [{"name": "...", "text": "..."}, ...]}
        """
        source = paper_dict.get("title", "unknown")
        chunks = []

        # Abstract always gets its own chunk
        if "abstract" in paper_dict:
            size = self._chunk_sizes.get("abstract", 250)
            chunks.extend(
                self._chunk_section(paper_dict["abstract"], source, "abstract", size)
            )

        for section in paper_dict.get("sections", []):
            name = section.get("name", "")
            text = section.get("text", "")
            key  = detect_section(name)
            size = self._chunk_sizes.get(key, self._chunk_sizes["default"])
            chunks.extend(self._chunk_section(text, source, key, size))

        logger.debug(f"[Chunker] '{source}' → {len(chunks)} chunks (dict mode)")
        return chunks

    # ── Private helpers ────────────────────────────────────────────────

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split raw text on markdown-style headers."""
        # Match # Header or ## Header or 1. Header or ALL CAPS HEADER
        pattern = re.compile(
            r"^(?:#{1,4}\s+|(?:\d+\.?\s+))(.+)$", re.MULTILINE
        )
        matches  = list(pattern.finditer(text))
        sections = []

        if not matches:
            return [("default", text)]

        # Text before first header
        if matches[0].start() > 0:
            sections.append(("introduction", text[:matches[0].start()]))

        for i, match in enumerate(matches):
            header = match.group(1).strip()
            start  = match.end()
            end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((header, text[start:end].strip()))

        return [(h, t) for h, t in sections if t.strip()]

    def _chunk_section(
        self, text: str, source: str, section: str, target_tokens: int
    ) -> list[dict]:
        """Split a section into overlapping chunks of ~target_tokens words."""
        words  = text.split()
        chunks = []
        start  = 0

        while start < len(words):
            end   = min(start + target_tokens, len(words))
            chunk_words = words[start:end]
            chunk_text  = " ".join(chunk_words)

            chunks.append({
                "id":          str(uuid.uuid4()),
                "text":        chunk_text,
                "source":      source,
                "section":     section,
                "token_count": len(chunk_words),
            })

            if end >= len(words):
                break
            start = end - self.overlap   # overlap with next chunk

        return chunks

    def _fixed_chunk(self, text: str, source: str, size: int) -> list[dict]:
        """Simple fixed-size chunking (used by baselines)."""
        words  = text.split()
        chunks = []
        start  = 0
        while start < len(words):
            end        = min(start + size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "id":          str(uuid.uuid4()),
                "text":        chunk_text,
                "source":      source,
                "section":     "default",
                "token_count": end - start,
            })
            if end >= len(words):
                break
            start = end - self.overlap
        return chunks
