import os
from modelscope.hub.snapshot_download import snapshot_download

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(PROJECT_ROOT, 'ai_models', 'modelscope_cache', 'models')

model_dir = snapshot_download('BAAI/bge-m3', cache_dir=cache_dir)
print(f"model downloaded to: {model_dir}")