"""
scripts/index_corpus.py
========================
Bulk-indexes a folder of scientific papers into ChromaDB.

Supports:
  - .txt files  (treated as plain text)
  - .json files (must follow {"title", "abstract", "sections"} schema)
  - .pdf files  (requires pypdf)

Usage:
    python scripts/index_corpus.py --corpus_path data/papers/ --config configs/config.yaml
    python scripts/index_corpus.py --corpus_path data/papers/ --mode fixed   # baseline chunks
    python scripts/index_corpus.py --clear   # wipe and rebuild index
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_paper_txt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {"title": path.stem, "abstract": "", "sections": [{"name": "default", "text": text}]}


def load_paper_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_paper_pdf(path: Path) -> dict:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text   = "\n".join(page.extract_text() or "" for page in reader.pages)
        return {"title": path.stem, "abstract": "", "sections": [{"name": "default", "text": text}]}
    except ImportError:
        logger.warning("pypdf not installed. Skipping PDF: %s", path)
        return None


def index_corpus(
    corpus_path: str,
    config_path: str = "configs/config.yaml",
    mode: str = "section_aware",
    clear: bool = False,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from src.layers.layer3_adaptation import RetrievalConfig
    from src.utils.hybrid_retriever import HybridRetriever
    from src.utils.chunker import SectionAwareChunker

    retrieval_config = RetrievalConfig(
        chunk_sizes = cfg["chunking"]["section_sizes"],
    )
    chunker = SectionAwareChunker(config=retrieval_config, overlap=cfg["chunking"]["overlap"])
    retriever = HybridRetriever(
        config          = retrieval_config,
        chroma_dir      = cfg["vector_store"]["persist_dir"],
        collection_name = cfg["vector_store"]["collection_name"],
        embed_model     = cfg["embedding"]["primary_model"],
    )

    if clear:
        logger.info("Clearing existing index...")
        retriever._chroma_client.delete_collection(cfg["vector_store"]["collection_name"])
        retriever._collection = retriever._chroma_client.get_or_create_collection(
            cfg["vector_store"]["collection_name"]
        )

    corpus_dir = Path(corpus_path)
    files = list(corpus_dir.rglob("*.txt")) + \
            list(corpus_dir.rglob("*.json")) + \
            list(corpus_dir.rglob("*.pdf"))

    logger.info(f"Found {len(files)} files in {corpus_path}")
    all_chunks = []

    for fp in files:
        try:
            if fp.suffix == ".txt":
                paper = load_paper_txt(fp)
            elif fp.suffix == ".json":
                paper = load_paper_json(fp)
            elif fp.suffix == ".pdf":
                paper = load_paper_pdf(fp)
            else:
                continue

            if paper is None:
                continue

            if mode == "section_aware":
                chunks = chunker.chunk_dict(paper)
            else:
                text   = paper.get("abstract","") + " " + \
                         " ".join(s["text"] for s in paper.get("sections",[]))
                chunks = chunker.chunk_text(text, source=paper["title"], mode="fixed")

            all_chunks.extend(chunks)
            logger.info(f"  {fp.name} → {len(chunks)} chunks")

        except Exception as e:
            logger.warning(f"  Failed to process {fp}: {e}")

    if all_chunks:
        # Limit corpus size on low-memory machines
        max_chunks = getattr(args, 'max_chunks', None)
        if max_chunks and len(all_chunks) > max_chunks:
            logger.warning(f"Truncating corpus to {max_chunks} chunks "
                           f"(was {len(all_chunks)}) — increase --max_chunks if RAM allows")
            all_chunks = all_chunks[:max_chunks]

        logger.info(f"Indexing {len(all_chunks)} total chunks...")
        retriever.add_documents(all_chunks)
        logger.info("Indexing complete.")
    else:
        logger.warning("No chunks to index. Check corpus_path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True, help="Folder containing paper files")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--mode",        default="section_aware", choices=["section_aware","fixed"])
    parser.add_argument("--clear",       action="store_true", help="Clear existing index first")
    parser.add_argument("--max_chunks", type=int, default=2000,
                        help="Max chunks to index (2000 recommended for 8 GB RAM)")
    args = parser.parse_args()

    index_corpus(args.corpus_path, args.config, args.mode, args.clear)
