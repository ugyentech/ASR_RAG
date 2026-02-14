# save this as test_query.py in the asr-rag folder
from src.asr_rag_pipeline import ASRRAGPipeline

with ASRRAGPipeline(config_path="configs/config.yaml") as pipeline:
    result = pipeline.query("What is the main contribution of this paper?")
    print("Answer:", result["answer"])
    print("Cycles used:", result["cycles_used"])