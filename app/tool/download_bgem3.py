import os
from modelscope.hub.snapshot_download import snapshot_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cache_dir = os.path.join(PROJECT_ROOT, "ai_models", "modelscope_cache", "models", "embeddings")

# check for config.json as the indicator that the model is fully downloaded
expected_config = os.path.join(cache_dir, "BAAI", "bge-m3", "config.json")
if os.path.exists(expected_config):
    print(f"Model already exists, skipping download: {cache_dir}")
else:
    model_dir = snapshot_download('BAAI/bge-m3', cache_dir=cache_dir)
    print(f"Model downloaded to: {model_dir}")