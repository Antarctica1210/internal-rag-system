from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

_LLM_MODE = os.getenv("LLM_MODE", "local")


def _env(key: str) -> str | None:
    """Pick the right env var based on LLM_MODE. 'online' prefers the _ALI variant."""
    if _LLM_MODE == "online":
        return os.getenv(f"{key}_ALI") or os.getenv(key)
    return os.getenv(key)


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    lv_model: str
    llm_model: str
    llm_temperature: float


lm_config = LLMConfig(
    base_url=_env("OPENAI_BASE_URL"),
    api_key=_env("OPENAI_API_KEY"),
    lv_model=_env("VL_MODEL"),
    llm_model=_env("LLM_DEFAULT_MODEL"),
    llm_temperature=float(_env("LLM_DEFAULT_TEMPERATURE") or 0.1),
)
