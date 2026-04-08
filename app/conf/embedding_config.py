# Import core dependencies: dataclass, environment variable loading, path handling
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load .env config file early (consistent with original code, only needs to run once)
load_dotenv()

# Define embedding config (covers all BGE-M3 settings)
@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # Local model path
    bge_m3: str       # Model repository identifier
    bge_device: str   # Compute device (cuda:0 / cpu)
    bge_fp16: bool    # Whether to enable half-precision (1=True / 0=False)

# Instantiate config object, consistent with the lm_config style in the original code
embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # Special handling: convert 1/0 from .env to bool, compatible with common numeric/string formats
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)
