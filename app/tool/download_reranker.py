import os
from modelscope.hub.snapshot_download import snapshot_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cache_dir = os.path.join(PROJECT_ROOT, "ai_models", "modelscope_cache", "models", "rerank")

expected_config = os.path.join(cache_dir, "BAAI", "bge-reranker-large", "config.json")
if os.path.exists(expected_config):
    print(f"Model already exists, skipping download: {cache_dir}")
else:
    snapshot_download(
        model_id="BAAI/bge-reranker-large",
        cache_dir=cache_dir,
    )
    print("Model downloaded to:", cache_dir)