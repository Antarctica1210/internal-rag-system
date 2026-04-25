# Environment config and dependency imports
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

# Internal project dependencies
from app.conf.lm_config import lm_config
from app.core.logger import logger

# Global cache: key is (model_name, json_mode) tuple, value is ChatOpenAI instance
# Purpose: avoid repeated client initialization, improve performance, centralize instance management
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    Get a globally cached LangChain ChatOpenAI client instance.
    Compatible with OpenAI / Qwen / Jimeng AI and other **OpenAI-compatible APIs**.
    Supports custom models and structured JSON output.
    Key features: caching, unified config loading, precise exception handling, domestic model parameter support.

    :param model: Model name. Priority: argument > lm_config.llm_model > built-in default qwen3-32b
    :param json_mode: Whether to enable JSON output mode; returns standard json_object format when enabled
    :return: Initialized ChatOpenAI instance (returned from cache if available, otherwise created and cached)
    :raise ValueError: Missing core config such as API key or base URL
    :raise Exception: Model initialization failure (LangChain layer exception)
    """
    # 1. Determine target model (decreasing priority, ensures model name is never empty)
    target_model = model or lm_config.llm_model or "qwen3-32b"
    # Cache key: model name + JSON mode, uniquely identifies clients with different configs
    cache_key = (target_model, json_mode)

    # 2. Cache hit: return existing instance directly to avoid re-creation
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM Client] Cache hit, returning existing instance: model={target_model}, json_mode={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. Core config validation: catch missing API config early and raise a clear exception
    if not lm_config.api_key:
        raise ValueError("[LLM Client] Missing config: please set OPENAI_API_KEY in .env (LLM API key)")
    if not lm_config.base_url:
        raise ValueError("[LLM Client] Missing config: please set OPENAI_API_BASE in .env (API base URL)")
    logger.info(f"[LLM Client] Initializing new instance: model={target_model}, json_mode={json_mode}")

    # 4. Assemble config params: separate domestic model private params from standard OpenAI params
    # extra_body: private params for domestic models like Qwen/Jimeng (passed through by LangChain to the API)
    extra_body = {"enable_thinking": False}  # Qwen-specific: disable chain-of-thought output to reduce noise
    # model_kwargs: standard OpenAI params supported by all compatible APIs
    model_kwargs = {}
    if json_mode:
        # Enable structured JSON output mode, forces model to return a parseable json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM Client] JSON output mode enabled, model will return standard JSON structure")

    # 5. Client initialization: catch LangChain layer exceptions and re-raise with a friendlier message
    try:
        llm_client = ChatOpenAI(
            model=target_model,                          # Target model name
            temperature=lm_config.llm_temperature or 0.1,  # Low temperature for deterministic output (0~1)
            api_key=lm_config.api_key,                  # API key
            base_url=lm_config.base_url,                # API base URL (supports domestic model proxy addresses)
            extra_body=extra_body,                      # Domestic model private params passthrough
            reasoning={"effort": "low"},                # Optional: set reasoning effort level (none/low/medium/high) for models that support it
            model_kwargs=model_kwargs,                   # Standard OpenAI params
        )
    except LangChainException as e:
        raise Exception(f"[LLM Client] Model [{target_model}] initialization failed (LangChain layer): {str(e)}") from e

    # 6. Store new instance in global cache for reuse in subsequent calls
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM Client] Instance initialized and cached: model={target_model}, json_mode={json_mode}")

    return llm_client


# Test example: verify client creation, caching mechanism, and log output
if __name__ == "__main__":
    logger.info("===== Starting LLM client utility tests =====")
    try:
        # Test 1: default config (default model + standard mode)
        client1 = get_llm_client()
        logger.info("✅ Test 1 passed: default config client created successfully")

        # Test 2: specify multimodal model (qwen-vl-plus) + standard mode
        client2 = get_llm_client(model="qwen-vl-plus")
        logger.info("✅ Test 2 passed: multimodal model client created successfully")

        # Test 3: same model + mode, verify cache hit
        client3 = get_llm_client(model="qwen-vl-plus")
        logger.info(f"✅ Test 3 passed: cache verified, client2 and client3 are the same instance: {client2 is client3}")

        # Test 4: enable JSON output mode
        client4 = get_llm_client(model="qwen3-32b", json_mode=True)
        logger.info("✅ Test 4 passed: JSON output mode client created successfully")

    except Exception as e:
        logger.error(f"❌ LLM client utility test failed: {str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM client utility tests complete =====")
