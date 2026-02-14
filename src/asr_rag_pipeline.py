"""
ASR-RAG Pipeline
=================
The main entry point. Wires together all three layers:

  Layer 1  →  Layer 2  →  Layer 3
  Self-Repair  Feedback    Adaptation

Usage:
    from src.asr_rag_pipeline import ASRRAGPipeline

    pipeline = ASRRAGPipeline(config_path="configs/config.yaml")
    result   = pipeline.query("What attention mechanism does the Transformer use?")
    print(result["answer"])

    # User provides feedback
    pipeline.user_rates(result["query_id"], rating=4)
    pipeline.user_corrects(result["query_id"], "The paper uses multi-head attention, not single-head")
"""

from __future__ import annotations
import logging
import uuid
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ASRRAGPipeline:
    """
    Full ASR-RAG pipeline with all three layers active.

    Parameters
    ----------
    config_path : str  — path to configs/config.yaml
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = self._load_config(config_path)
        self._setup_logging()

        logger.info("Initialising ASR-RAG Pipeline...")

        # ── Shared retrieval config (Layer 3 updates this live) ────────
        from src.layers.layer3_adaptation import RetrievalConfig
        self.retrieval_config = RetrievalConfig(
            alpha_dense  = self.cfg["retrieval"]["alpha_dense"],
            alpha_sparse = self.cfg["retrieval"]["alpha_sparse"],
            top_k        = self.cfg["retrieval"]["top_k"],
            chunk_sizes  = self.cfg["chunking"]["section_sizes"],
        )

        # ── LLM ───────────────────────────────────────────────────────
        self.llm = self._load_llm()

        # ── Layer 1: Self-Repair ───────────────────────────────────────
        from src.utils.hybrid_retriever import HybridRetriever
        from src.layers.layer1_self_repair import SelfRepairModule

        self.retriever = HybridRetriever(
            config          = self.retrieval_config,
            chroma_dir      = self.cfg["vector_store"]["persist_dir"],
            collection_name = self.cfg["vector_store"]["collection_name"],
            embed_model     = self.cfg["embedding"]["primary_model"],
        )

        self.self_repair = SelfRepairModule(
            retriever           = self.retriever,
            llm                 = self.llm,
            relevance_thresh    = self.cfg["self_repair"]["relevance_threshold"],
            consistency_thresh  = self.cfg["self_repair"]["consistency_threshold"],
            max_iterations      = self.cfg["self_repair"]["max_iterations"],
        )

        # ── Layer 2: Feedback Collection ──────────────────────────────
        from src.layers.layer2_feedback import FeedbackCollector
        self.feedback = FeedbackCollector(
            db_path = self.cfg["feedback"]["db_path"]
        )

        # ── Layer 3: Continuous Adaptation ────────────────────────────
        from src.layers.layer3_adaptation import AdaptationEngine
        self.adaptation = AdaptationEngine(
            config               = self.retrieval_config,
            feedback_collector   = self.feedback,
            adaptation_interval  = self.cfg["adaptation"]["interval"],
            config_save_path     = str(Path(self.cfg["paths"]["results_dir"]) / "config_snapshots"),
            success_rate_threshold = self.cfg["adaptation"]["success_rate_threshold"],
        )

        # Start background adaptation thread
        self.adaptation.start()
        logger.info("ASR-RAG Pipeline ready.")

    # ── Public API ─────────────────────────────────────────────────────

    def query(self, question: str, query_id: Optional[str] = None) -> dict:
        """
        Process a user question through all three layers.

        Returns
        -------
        dict with keys:
          - query_id       : str
          - question       : str
          - answer         : str
          - context        : list[dict]  — retrieved passages
          - cycles_used    : int
          - relevance_score: float
          - consistency_score: float
          - success        : bool
        """
        if query_id is None:
            query_id = str(uuid.uuid4())[:8]

        logger.info(f"[Pipeline] Query {query_id}: '{question[:80]}'")

        # Layer 1: Self-repair
        result = self.self_repair.process(question, query_id=query_id)

        # Layer 2: Log implicit feedback
        self.feedback.record(result.feedback)

        return {
            "query_id":          query_id,
            "question":          question,
            "answer":            result.answer,
            "context":           result.context,
            "cycles_used":       result.feedback.cycles_used,
            "relevance_score":   result.feedback.relevance_score,
            "consistency_score": result.feedback.consistency_score,
            "success":           result.success,
        }

    def index_documents(self, docs: list[dict]):
        """
        Add documents to the vector store.
        Use scripts/index_corpus.py for bulk indexing.
        """
        self.retriever.add_documents(docs)

    def index_paper(self, paper: dict):
        """
        Index a structured paper dict using section-aware chunking.
        paper = {"title": "...", "abstract": "...", "sections": [...]}
        """
        from src.utils.chunker import SectionAwareChunker
        chunker = SectionAwareChunker(config=self.retrieval_config)
        chunks  = chunker.chunk_dict(paper)
        self.retriever.add_documents(chunks)
        logger.info(f"[Pipeline] Indexed paper '{paper.get('title','')}' "
                    f"→ {len(chunks)} chunks")

    # ── Explicit feedback API ──────────────────────────────────────────

    def user_rates(self, query_id: str, rating: int):
        """User rates a response 1–5. Call from your UI."""
        self.feedback.user_rates(query_id, rating)

    def user_corrects(self, query_id: str, correction: str):
        """User provides a correction. Call from your UI."""
        self.feedback.user_corrects(query_id, correction)

    def user_rates_passage(self, query_id: str, ratings: dict[str, int]):
        """User rates individual passages {doc_id: 0|1}."""
        self.feedback.user_rates_passage(query_id, ratings)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def shutdown(self):
        """Gracefully stop the adaptation engine before exit."""
        self.adaptation.stop()
        logger.info("[Pipeline] Shut down cleanly.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _load_llm(self):
        """Load Llama-3-8B via HuggingFace pipeline wrapped in LangChain."""
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            model_name = self.cfg["model"]["name"]
            logger.info(f"Loading LLM: {model_name}")

            quant_cfg = None
            if self.cfg["model"]["quantization"] == "4bit":
                from transformers import BitsAndBytesConfig
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model     = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = quant_cfg,
                device_map          = "auto",
                torch_dtype         = torch.bfloat16,
            )
            pipe = pipeline(
                "text-generation",
                model           = model,
                tokenizer       = tokenizer,
                max_new_tokens  = self.cfg["model"]["max_new_tokens"],
                temperature     = self.cfg["model"]["temperature"],
                do_sample       = True,
                return_full_text= False,
            )
            return HuggingFacePipeline(pipeline=pipe)

        except ImportError as e:
            logger.warning(f"Could not load full LLM ({e}). Using mock LLM for testing.")
            return self._mock_llm()

    @staticmethod
    def _mock_llm():
        """Lightweight mock LLM for unit tests and CI environments."""
        class MockLLM:
            def invoke(self, prompt):
                class R:
                    content = "Mock answer: the retrieved context provides the relevant information."
                return R()
        return MockLLM()

    @staticmethod
    def _setup_logging():
        logging.basicConfig(
            level   = logging.INFO,
            format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt = "%H:%M:%S",
        )
