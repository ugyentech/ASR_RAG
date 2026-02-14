# ASR-RAG — Adaptive Self-Repairing RAG

<div align="center">
    
**Continuous Learning Through Real-World Feedback and Iterative Self-Correction**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square&logo=chainlink)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat-square)](https://trychroma.com)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

<br/>

> Most RAG systems fail the same way twice.  
> **ASR-RAG learns from every failure so it never has to.**

<br/>

[**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**Results**](#-results) · [**Contributing**](#-contributing) · [**Paper**](#-paper)

</div>

---

## 🤔 Why ASR-RAG?

Standard RAG pipelines are **static**. They retrieve, generate, and move on — making the same mistakes tomorrow that they made today. ASR-RAG fixes this with three tightly integrated layers:

| Problem with vanilla RAG | How ASR-RAG solves it |
|---|---|
| Retrieves irrelevant passages | Layer 1 scores relevance and reformulates the query in real-time |
| Hallucinates claims not in context | Layer 1 self-critiques and regenerates with strict grounding |
| Never improves between sessions | Layer 2 captures every failure; Layer 3 adapts the retrieval config |
| Requires retraining to get better | Adaptation happens via config updates — **no model retraining** |

---

## ✨ Key Features

- **🔧 Real-Time Self-Repair** — Per-query iterative correction with configurable relevance (`τ_r`) and consistency (`τ_c`) thresholds
- **📊 Dual Feedback Signals** — Implicit signals (reformulation counts, cycle patterns) + explicit signals (user corrections, ratings)
- **⚡ Hybrid Retrieval** — Dense vector search + BM25 sparse retrieval with adaptive weight tuning
- **🧠 Section-Aware Chunking** — Paper sections (Abstract, Methods, Results) get different chunk sizes for optimal scientific text retrieval
- **🔄 Zero-Downtime Adaptation** — Hot-swap config updates every Δt queries — live system never pauses
- **📈 Compounds Over Time** — Performance improves across adaptation cycles without touching model weights

---

## 📊 Results

Evaluated on **QASPER** + **SciDocs-RAG** (1,000 scientific QA queries) against four baselines:

| System | Answer Relevancy | Faithfulness | Context Precision | Context Recall |
|--------|:-:|:-:|:-:|:-:|
| Vanilla RAG | 0.718 | 0.803 | 0.443 | 0.371 |
| Self-RAG | 0.748 | 0.823 | 0.472 | 0.391 |
| FLARE | 0.756 | 0.831 | 0.481 | 0.398 |
| MemPrompt | 0.764 | 0.842 | 0.493 | 0.412 |
| **ASR-RAG (ours)** | **0.835** | **0.908** | **0.582** | **0.530** |
| Δ vs. best baseline | **+9.3%** | **+7.8%** | **+18.1%** | **+28.6%** |

> Hallucination rate reduced from **8.6% → 2.5%** (70.9% reduction) over 5 adaptation cycles.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│         Layer 1 — Real-Time Self-Repair Module          │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Hybrid  │→ │Relevance │→ │  Query   │→ │  Self  │  │
│  │Retrieval │  │ Scoring  │  │ Reform.  │  │Critique│  │
│  │Dense+BM25│  │τ_r = 0.6 │  │          │  │τ_c=0.7 │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │  feedback
┌──────────────────────▼──────────────────────────────────┐
│         Layer 2 — Feedback Collection & Analysis        │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   Implicit Signals   │  │    Explicit Signals      │ │
│  │ • Query reformulations│  │ • Retrieval cycles   │  │
│  │ • Answer rejections  │  │ • Relevance ratings      │ │
│  │ • Flagged hallucinations │  │ • User corrections       │ │
│  └──────────────────────┘  └──────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │  patterns
┌──────────────────────▼──────────────────────────────────┐
│         Layer 3 — Continuous Adaptation Engine          │
│                                                         │
│  Hybrid Weights · Embeddings · Chunk Boundaries        │
│  Reranking · Query Expansion Vocabulary                 │
│                                                         │
│  (Every Δt=200 queries · async · zero downtime)        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Generated Response
```

---

## 📁 Project Structure

```
asr-rag/
│
├── src/
│   ├── asr_rag_pipeline.py          # 🔌 Main pipeline — start here
│   ├── layers/
│   │   ├── layer1_self_repair.py    # Algorithm 1: iterative self-repair
│   │   ├── layer2_feedback.py       # SQLite feedback collection
│   │   └── layer3_adaptation.py     # Algorithm 2: async adaptation engine
│   └── utils/
│       ├── hybrid_retriever.py      # Dense (ChromaDB) + sparse (BM25)
│       ├── query_expander.py        # Scientific term expansion
│       ├── consistency_checker.py   # LLM faithfulness scoring
│       └── chunker.py               # Section-aware paper chunking
│
├── evaluation/
│   ├── ragas_evaluator.py           # RAGAS metric computation
│   └── baselines.py                 # Vanilla RAG, Self-RAG, FLARE, MemPrompt
│
├── experiments/
│   ├── run_qasper.py                # QASPER benchmark runner
│   └── run_scidocs.py               # SciDocs-RAG benchmark runner
│
├── tests/
│   ├── test_layer1.py
│   ├── test_layer2.py
│   └── test_layer3.py
│
├── scripts/
│   ├── index_corpus.py              # Bulk index PDF/TXT/JSON papers
│   └── run_evaluation.sh            # End-to-end eval script
│
└── configs/
    └── config.yaml                  # All hyperparameters in one place
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone git@github.com:ugyentech/ASR_RAG.git
cd ASR_RAG
pip install -r requirements.txt
```

> **GPU recommended** (4-bit quantisation requires CUDA 12.1+).  
> CPU inference works but is slow. Set `quantization: null` in `config.yaml` for CPU.

### 2. Configure

Open `configs/config.yaml` and set your model and paths:

```yaml
model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"  # or any HuggingFace model
  quantization: "4bit"

vector_store:
  persist_dir: "data/chroma_db"
```

### 3. Index your papers

```bash
# Index a folder of PDFs, JSONs, or TXTs
python scripts/index_corpus.py --corpus_path data/papers/

# Use fixed chunking instead of section-aware (for non-paper text)
python scripts/index_corpus.py --corpus_path data/papers/ --mode fixed
```

### 4. Ask a question

```python
from src.asr_rag_pipeline import ASRRAGPipeline

with ASRRAGPipeline(config_path="configs/config.yaml") as pipeline:

    result = pipeline.query("What attention mechanism does the Transformer use?")
    print(result["answer"])
    print(f"Cycles used: {result['cycles_used']}")
    print(f"Relevance:   {result['relevance_score']:.2f}")
    print(f"Faithfulness:{result['consistency_score']:.2f}")

    # Provide feedback (optional but improves future queries)
    pipeline.user_rates(result["query_id"], rating=4)
    pipeline.user_corrects(result["query_id"], "The paper uses multi-head attention, not single-head")
```

### 5. Run the full evaluation

```bash
bash scripts/run_evaluation.sh
```

Results are saved to `results/qasper/` and `results/scidocs/`.

---

## ⚙️ Configuration Reference

All settings live in `configs/config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `self_repair.relevance_threshold` | `0.60` | τ_r — triggers query reformulation |
| `self_repair.consistency_threshold` | `0.70` | τ_c — triggers constrained regeneration |
| `self_repair.max_iterations` | `3` | N — max repair cycles per query |
| `retrieval.alpha_dense` | `0.70` | Weight for dense vector search |
| `retrieval.alpha_sparse` | `0.30` | Weight for BM25 sparse retrieval |
| `adaptation.interval` | `200` | Δt — queries between adaptation cycles |
| `chunking.strategy` | `section_aware` | `section_aware` or `fixed` |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific layer's tests
pytest tests/test_layer1.py -v
```

---

## 🗺️ Roadmap

- [ ] **v1.1** — Reinforcement learning integration for query expansion optimisation
- [ ] **v1.2** — Multi-modal support (figures, tables, equations in papers)
- [ ] **v1.3** — Federated adaptation across multiple deployments
- [ ] **v2.0** — Web UI with real-time feedback dashboard
- [ ] **v2.1** — REST API for production deployment

---

## 🤝 Contributing

Contributions are very welcome! Here's how to get involved:

### Ways to Contribute

| Type | Description |
|------|-------------|
| 🐛 **Bug Reports** | Found something broken? Open an issue with a minimal reproducible example |
| 💡 **Feature Requests** | Have an idea? Open an issue describing the use case |
| 🔧 **Code** | Pick an open issue, or propose your own improvement |
| 📖 **Documentation** | Improve docstrings, add examples, fix typos |
| 🧪 **Tests** | Add test coverage for untested areas |
| 📊 **Experiments** | Run evaluations on new datasets and share results |

### Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/asr-rag.git
cd asr-rag

# 2. Create a branch for your change
git checkout -b feature/your-feature-name

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Make your changes, then run tests
pytest tests/

# 5. Commit and push
git add .
git commit -m "feat: describe your change clearly"
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub
```

### Contribution Guidelines

- **Code style**: Follow PEP 8. We use `black` for formatting — run `black src/` before committing.
- **Docstrings**: Every public function/class needs a docstring explaining what it does and its parameters.
- **Tests**: New features should come with tests in `tests/`. Bug fixes should include a regression test.
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.
- **PR size**: Keep pull requests focused on one thing. Smaller PRs are reviewed faster.

### Good First Issues

New to the codebase? Look for issues labelled `good first issue` — these are small, well-defined tasks perfect for getting familiar with the code.

---

## 📋 Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| CUDA (recommended) | 12.1+ |
| RAM | 16 GB minimum, 32 GB recommended |
| VRAM (GPU) | 16 GB for 4-bit quantised Llama-3-8B |

---

## 🙏 Acknowledgements

This work builds on several excellent open-source projects and research papers:

- [**Self-RAG**](https://github.com/AkariAsai/self-rag) — Asai et al., 2023
- [**FLARE**](https://github.com/jzbjyb/FLARE) — Jiang et al., 2023
- [**MemPrompt**](https://github.com/madaan/memprompt) — Madaan et al., 2022
- [**RAGAS**](https://github.com/explodinggradients/ragas) — Es et al., 2023
- [**LangChain**](https://github.com/langchain-ai/langchain)
- [**ChromaDB**](https://github.com/chroma-core/chroma)
- [**QASPER**](https://allenai.org/data/qasper) — Dasigi et al., 2021
- [**SciDocs / BEIR**](https://github.com/beir-cellar/beir) — Thakur et al., 2021

---

## 📄 Paper

If you use ASR-RAG in your research, please cite:

```bibtex
@article{asrrag2025,
  title   = {Adaptive Self-Repairing RAG: Continuous Learning Through Real-World Feedback and Iterative Self-Correction},
  author  = {[Ugyen Dendup]},
  year    = {2026},
  url     = {git@github.com:ugyentech/ASR_RAG.git}
}
```

---

## 📬 Contact

Have a question that isn't covered here?

- **Open an issue** — for bugs, features, or general questions about the code
- **Start a Discussion** — for broader ideas, research questions, or showcasing what you've built with ASR-RAG

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

You are free to use, modify, and distribute this software, including for commercial purposes, as long as you include the original license notice.

---

<div align="center">

Made with ❤️ — contributions welcome!

⭐ **Star this repo** if you find it useful — it helps others discover the project.

</div>
