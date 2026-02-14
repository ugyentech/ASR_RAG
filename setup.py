from setuptools import setup, find_packages

setup(
    name="asr-rag",
    version="1.0.0",
    description="Adaptive Self-Repairing RAG: Continuous Learning Through Real-World Feedback",
    author="[Author Name]",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.2.0",
        "sentence-transformers>=2.7.0",
        "chromadb>=0.5.0",
        "rank_bm25>=0.2.2",
        "ragas>=0.1.7",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
    ],
)
