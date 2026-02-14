"""
SciDocs-RAG Benchmark Experiment
==================================
Runs ASR-RAG evaluation on SciDocs-RAG (BEIR split):
  - 500 cross-paper scientific queries
  - Multi-document retrieval (2-5 papers per query)
  - Same 4 baselines as QASPER experiment

Usage:
    python experiments/run_scidocs.py
    python experiments/run_scidocs.py --system asr_rag
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


def load_scidocs_rag(max_queries: int = 500) -> tuple[list[dict], list[dict]]:
    """
    Load SciDocs-RAG from BEIR benchmark.
    Returns (docs, queries).

    docs    : list of {id, title, text, source}
    queries : list of {question, ground_truth, relevant_doc_ids}
    """
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
        data_path = util.download_and_unzip(url, "data/beir/")
        corpus, queries_raw, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        docs = [
            {
                "id":     doc_id,
                "title":  doc.get("title", ""),
                "text":   doc.get("text", ""),
                "source": doc.get("title", doc_id),
                "section": "default",
            }
            for doc_id, doc in corpus.items()
        ]

        queries = []
        for qid, question in queries_raw.items():
            relevant_ids = list(qrels.get(qid, {}).keys())
            queries.append({
                "question":        question,
                "ground_truth":    "",  # SciDocs uses passage retrieval, not text answers
                "relevant_doc_ids": relevant_ids,
            })
            if len(queries) >= max_queries:
                break

        return docs, queries[:max_queries]

    except Exception as e:
        logger.warning(f"Could not load SciDocs-RAG ({e}). Using synthetic data.")
        return _synthetic_scidocs(max_queries)


def _synthetic_scidocs(n: int) -> tuple[list[dict], list[dict]]:
    """Minimal synthetic multi-doc data for offline testing."""
    docs = [
        {"id": "bert", "title": "BERT", "source": "BERT",
         "text": "BERT uses a bidirectional transformer encoder pre-trained with masked language modelling and next sentence prediction.", "section": "abstract"},
        {"id": "gpt2", "title": "GPT-2", "source": "GPT-2",
         "text": "GPT-2 uses a unidirectional transformer decoder pre-trained with a causal language modelling objective on 40GB of internet text.", "section": "abstract"},
        {"id": "roberta", "title": "RoBERTa", "source": "RoBERTa",
         "text": "RoBERTa improves on BERT by removing next sentence prediction, training on more data, and using dynamic masking.", "section": "abstract"},
        {"id": "t5", "title": "T5", "source": "T5",
         "text": "T5 frames every NLP task as a text-to-text problem and pre-trains a unified encoder-decoder transformer.", "section": "abstract"},
    ]
    queries = [
        {"question": "How does BERT differ from GPT-2 in pre-training objectives?",
         "ground_truth": "BERT uses bidirectional masked LM; GPT-2 uses causal LM.",
         "relevant_doc_ids": ["bert", "gpt2"]},
        {"question": "What improvements does RoBERTa make over BERT?",
         "ground_truth": "Removes NSP, more data, dynamic masking.",
         "relevant_doc_ids": ["bert", "roberta"]},
        {"question": "How does T5 unify NLP tasks?",
         "ground_truth": "By treating every task as text-to-text.",
         "relevant_doc_ids": ["t5"]},
    ] * (n // 3 + 1)

    return docs, queries[:n]


def run_experiment(
    system_name: str,
    pipeline_or_baseline,
    queries: list[dict],
    warmup: int,
    record_every: int,
    results_dir: Path,
) -> list[dict]:
    """Run one system on SciDocs-RAG queries and record temporal metrics."""
    from evaluation.ragas_evaluator import RagasEvaluator

    evaluator   = RagasEvaluator(results_dir=str(results_dir / system_name))
    eval_queries = queries[warmup:]

    timeline = []
    samples  = []

    logger.info(f"\n{'='*60}\nRunning: {system_name.upper()} on SciDocs-RAG\n{'='*60}")

    for i, q in enumerate(eval_queries, 1):
        result = pipeline_or_baseline.query(q["question"])

        samples.append({
            "question":     q["question"],
            "answer":       result["answer"],
            "contexts":     [d["text"] for d in result.get("context", [])],
            "ground_truth": q.get("ground_truth", ""),
        })

        if i % record_every == 0 or i == len(eval_queries):
            batch  = samples[max(0, i - record_every):]
            scores = evaluator.evaluate(
                batch,
                run_name=f"{system_name}_cycle{i // record_every}",
                save=True,
            )
            timeline.append({
                "system":            system_name,
                "cycle":             i // record_every,
                "queries_processed": i,
                **scores,
            })

    out_path = results_dir / f"{system_name}_timeline.json"
    with open(out_path, "w") as f:
        json.dump(timeline, f, indent=2)
    logger.info(f"Timeline saved → {out_path}")
    return timeline


def main():
    parser = argparse.ArgumentParser(description="Run SciDocs-RAG experiment")
    parser.add_argument("--system",      default="all",
                        choices=["all", "asr_rag", "vanilla", "self_rag", "flare", "memprompt"])
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--max_queries", type=int, default=500)
    parser.add_argument("--warmup",      type=int, default=100)
    parser.add_argument("--record_every",type=int, default=200)
    parser.add_argument("--results_dir", default="results/scidocs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SciDocs-RAG dataset...")
    docs, queries = load_scidocs_rag(max_queries=args.max_queries)
    logger.info(f"Loaded {len(docs)} documents, {len(queries)} queries")

    from src.asr_rag_pipeline import ASRRAGPipeline
    pipeline = ASRRAGPipeline(config_path=args.config)

    logger.info("Indexing documents...")
    pipeline.index_documents(docs)

    logger.info(f"Warm-up phase: {args.warmup} queries...")
    for q in queries[:args.warmup]:
        pipeline.query(q["question"])

    all_timelines = {}

    systems = {
        "asr_rag":  pipeline,
    }

    if args.system in ("all", "vanilla"):
        from evaluation.baselines import VanillaRAG
        systems["vanilla"] = VanillaRAG(pipeline.retriever, pipeline.llm)

    if args.system in ("all", "self_rag"):
        from evaluation.baselines import SelfRAG
        systems["self_rag"] = SelfRAG(pipeline.retriever, pipeline.llm)

    if args.system in ("all", "flare"):
        from evaluation.baselines import FLARE
        systems["flare"] = FLARE(pipeline.retriever, pipeline.llm)

    if args.system in ("all", "memprompt"):
        from evaluation.baselines import MemPrompt
        systems["memprompt"] = MemPrompt(pipeline.retriever, pipeline.llm)

    for name, system in systems.items():
        if args.system == "all" or args.system == name:
            all_timelines[name] = run_experiment(
                name, system, queries, args.warmup, args.record_every, results_dir
            )

    combined_path = results_dir / "all_systems_timeline.json"
    with open(combined_path, "w") as f:
        json.dump(all_timelines, f, indent=2)
    logger.info(f"\nAll results saved → {results_dir}")

    pipeline.shutdown()


if __name__ == "__main__":
    main()
