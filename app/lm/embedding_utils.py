from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# Singleton model instance to avoid repeated initialization
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    Get the BGE-M3 model singleton, automatically loading config from environment variables.
    :return: An initialized BGEM3EmbeddingFunction instance
    """
    global _bge_m3_ef
    # Singleton pattern: return existing instance directly to avoid reloading the model
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3 model singleton already exists, returning existing instance")
        return _bge_m3_ef

    # Load config from environment variables, fall back to defaults if not set
    # Use a local path if available! Otherwise "BAAI/bge-m3" will be downloaded automatically. A URL can also be used for cloud deployments.
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # Log model initialization config for troubleshooting
    logger.info(
        "Starting BGE-M3 model initialization",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # Initialize BGE-M3 model with native L2 normalization enabled (compatible with Milvus IP inner-product search)
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # Native L2 normalization applied to both dense and sparse vectors
        )
        logger.success("BGE-M3 model initialized successfully with native L2 normalization enabled")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3 model initialization failed: {str(e)}", exc_info=True)
        raise  # Re-raise to let the caller handle the exception


def generate_embeddings(texts):
    """
    Generate dense + sparse hybrid vector embeddings for a list of texts (with native L2 normalization).
    :param texts: List of texts to embed; a single text must also be wrapped in a list
    :return: Dict with keys 'dense' and 'sparse', containing nested lists and list of dicts respectively
    :raise: Any exception raised during embedding generation, to be handled by the caller
    """
    # Validate input
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("Invalid input for embedding generation: texts must be a non-empty list")
        raise ValueError("Parameter texts must be a non-empty list containing text strings")

    logger.info(f"Starting hybrid vector embedding generation for {len(texts)} texts")
    try:
        # Load BGE-M3 model singleton
        model = get_bge_m3_ef()
        # Encode documents to produce dense vectors + sparse vectors (CSR format)
        embeddings = model.encode_documents(texts)
        logger.debug(f"Model encoding complete, parsing sparse vector format for {len(texts)} entries")

        # Initialize sparse vector processing results, parsed into dict format (for serialization/storage)
        processed_sparse = []
        for i in range(len(texts)):
            # Extract sparse vector indices for the i-th text: np.int64 → Python int (required for hashable dict keys)
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Extract sparse vector weights for the i-th text: np.float32 → Python float (for JSON serialization / API responses)
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Build {feature_index: normalized_weight} sparse vector dict
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # Build final result: convert dense vectors to lists (resolves numpy array serialization issue)
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # Nested list, one entry per input text
            "sparse": processed_sparse  # List of dicts, L2-normalized by the model
        }
        logger.success(f"Vector embedding generation complete for {len(texts)} texts, output format is production-ready")
        return result

    except Exception as e:
        logger.error(f"Text vector generation failed: {str(e)}", exc_info=True)
        raise  # Do not swallow the exception; propagate to caller for retry/fallback handling


"""
Key design highlights and compatibility notes:
1. Native model normalization: normalize_embeddings=True applies L2 normalization to both dense and sparse vectors,
   fully compatible with Milvus IP inner-product search (unit vectors make IP equivalent to cosine, faster computation);
2. NumPy key issue resolved: .tolist() on sparse_indices converts np.int64 to Python int,
   satisfying the hashable key requirement for dicts with no risk of errors;
3. Sparse value serialization: .tolist() on sparse_data converts np.float32 to Python float,
   supporting JSON writing, API responses, and Milvus insertion in all scenarios;
4. Singleton optimization: the model is initialized only once, avoiding costly repeated loading
   and improving batch processing efficiency;
5. Output format matches business usage: returns dense as nested list and sparse as list of dicts,
   compatible with vector_result["dense"][0] / sparse_vector["sparse"][0] access patterns;
6. Tiered logging coverage: full logging from model initialization and vector generation to error reporting,
   enabling production-environment troubleshooting;
7. Input validation: guards against empty list or non-list inputs that would cause internal errors,
   improving utility class robustness.
"""
