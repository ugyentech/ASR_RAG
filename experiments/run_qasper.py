"""
QASPER Benchmark Experiment
=============================
Runs the full ASR-RAG evaluation on QASPER:
  - Loads the QASPER test split (500 queries)
  - Indexes all source papers with section-aware chunking
  - Runs ASR-RAG + all 4 baselines
  - Records metrics every 200 queries (5 adaptation cycles)
  - Saves results to results/qasper/

Usage:
    python experiments/run_qasper.py
    python experiments/run_qasper.py --system all          # run all systems
    python experiments/run_qasper.py --system asr_rag      # run one system
    python experiments/run_qasper.py --warmup 100          # warm-up query count
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


def load_qasper(split: str = "test", max_queries: int = 500) -> tuple[list[dict], list[dict]]:
    """
    Load QASPER dataset.
    Returns (papers, queries).

    papers  : list of {title, abstract, sections: [{name, text}]}
    queries : list of {question, ground_truth, paper_id}
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("allenai/qasper", split=split)
    except Exception as e:
        logger.warning(f"Could not load QASPER from HuggingFace ({e}). Using synthetic data.")
        return _synthetic_qasper(max_queries)

    papers  = []
    queries = []

    for item in dataset:
        paper = {
            "title":    item.get("title", ""),
            "abstract": item.get("abstract", ""),
            "sections": [
                {"name": s["section_name"], "text": " ".join(s["paragraphs"])}
                for s in item.get("full_text", [])
            ],
        }
        papers.append(paper)

        for qa in item.get("qas", []):
            question = qa.get("question", "")
            answers  = qa.get("answers", [])
            ground_truth = answers[0]["answer"][0].get("free_form_answer", "") if answers else ""

            if question and ground_truth:
                queries.append({
                    "question":     question,
                    "ground_truth": ground_truth,
                    "paper_title":  paper["title"],
                })

            if len(queries) >= max_queries:
                break
        if len(queries) >= max_queries:
            break

    return papers, queries[:max_queries]


def _synthetic_qasper(n: int) -> tuple[list[dict], list[dict]]:
    """Minimal synthetic data for testing without internet access."""
    papers = [
        {
            "title": "Attention Is All You Need",
            "abstract": "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms.",
            "sections": [
                {"name": "Introduction", "text": "Recurrent neural networks have been the state of the art for sequence modelling tasks."},
                {"name": "Model Architecture", "text": "The Transformer uses multi-head self-attention with positional encodings. The model consists of an encoder and decoder, each with 6 layers."},
                {"name": "Results", "text": "On the WMT 2014 English-to-German translation task, the Transformer achieves 28.4 BLEU, outperforming all previously published models."},
            ],
        }
    ]
    queries = [
        {"question": "What is the main contribution of the Transformer?",
         "ground_truth": "A novel architecture based solely on attention mechanisms.",
         "paper_title": "Attention Is All You Need"},
        {"question": "How many layers does the Transformer encoder have?",
         "ground_truth": "6 layers.",
         "paper_title": "Attention Is All You Need"},
        {"question": "What BLEU score did the Transformer achieve on WMT 2014?",
         "ground_truth": "28.4 BLEU on English-to-German.",
         "paper_title": "Attention Is All You Need"},
    ] * (n // 3 + 1)

    return papers, queries[:n]


def run_experiment(
    system_name: str,
    pipeline_or_baseline,
    queries: list[dict],
    warmup: int,
    record_every: int,
    results_dir: Path,
) -> list[dict]:
    """Run one system on all queries and record temporal metrics."""
    from evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(results_dir=str(results_dir / system_name))
    eval_queries = queries[warmup:]   # skip warm-up for metrics

    timeline = []
    samples  = []

    logger.info(f"\n{'='*60}\nRunning: {system_name.upper()}\nQueries: {len(eval_queries)}\n{'='*60}")

    for i, q in enumerate(eval_queries, 1):
        result = pipeline_or_baseline.query(q["question"])

        samples.append({
            "question":     q["question"],
            "answer":       result["answer"],
            "contexts":     [d["text"] for d in result.get("context", [])],
            "ground_truth": q["ground_truth"],
        })

        if i % record_every == 0 or i == len(eval_queries):
            batch   = samples[max(0, i - record_every):]
            scores  = evaluator.evaluate(
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

    # Save full timeline
    out_path = results_dir / f"{system_name}_timeline.json"
    with open(out_path, "w") as f:
        json.dump(timeline, f, indent=2)
    logger.info(f"Timeline saved → {out_path}")

    return timeline


def main():
    parser = argparse.ArgumentParser(description="Run QASPER experiment")
    parser.add_argument("--system",      default="all",
                        choices=["all", "asr_rag", "vanilla", "self_rag", "flare", "memprompt"])
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--max_queries", type=int, default=500)
    parser.add_argument("--warmup",      type=int, default=100)
    parser.add_argument("--record_every",type=int, default=200)
    parser.add_argument("--results_dir", default="results/qasper")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────
    logger.info("Loading QASPER dataset...")
    papers, queries = load_qasper(max_queries=args.max_queries)
    logger.info(f"Loaded {len(papers)} papers, {len(queries)} queries")

    # ── Build ASR-RAG pipeline ─────────────────────────────────────────
    from src.asr_rag_pipeline import ASRRAGPipeline
    pipeline = ASRRAGPipeline(config_path=args.config)

    # Index papers with section-aware chunking
    logger.info("Indexing papers...")
    for paper in papers:
        pipeline.index_paper(paper)

    # Warm-up: process warm-up queries without recording metrics
    logger.info(f"Warm-up phase: {args.warmup} queries...")
    for q in queries[:args.warmup]:
        pipeline.query(q["question"])

    # ── Run selected systems ───────────────────────────────────────────
    all_timelines = {}

    if args.system in ("all", "asr_rag"):
        all_timelines["asr_rag"] = run_experiment(
            "asr_rag", pipeline, queries, args.warmup, args.record_every, results_dir
        )

    if args.system in ("all", "vanilla"):
        from evaluation.baselines import VanillaRAG
        vanilla = VanillaRAG(pipeline.retriever, pipeline.llm)
        all_timelines["vanilla"] = run_experiment(
            "vanilla", vanilla, queries, args.warmup, args.record_every, results_dir
        )

    if args.system in ("all", "self_rag"):
        from evaluation.baselines import SelfRAG
        self_rag = SelfRAG(pipeline.retriever, pipeline.llm)
        all_timelines["self_rag"] = run_experiment(
            "self_rag", self_rag, queries, args.warmup, args.record_every, results_dir
        )

    if args.system in ("all", "flare"):
        from evaluation.baselines import FLARE
        flare = FLARE(pipeline.retriever, pipeline.llm)
        all_timelines["flare"] = run_experiment(
            "flare", flare, queries, args.warmup, args.record_every, results_dir
        )

    if args.system in ("all", "memprompt"):
        from evaluation.baselines import MemPrompt
        memprompt = MemPrompt(pipeline.retriever, pipeline.llm)
        all_timelines["memprompt"] = run_experiment(
            "memprompt", memprompt, queries, args.warmup, args.record_every, results_dir
        )

    # Save combined timeline
    combined_path = results_dir / "all_systems_timeline.json"
    with open(combined_path, "w") as f:
        json.dump(all_timelines, f, indent=2)
    logger.info(f"\nAll results saved → {results_dir}")

    pipeline.shutdown()


if __name__ == "__main__":
    main()
