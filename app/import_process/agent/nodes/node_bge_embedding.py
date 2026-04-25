import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_1_validate_input(state: ImportGraphState) -> List[Dict[str, Any]]:
    """
    Step 1: Validate input data before vectorisation.
    Extracts the chunks list from state and raises ValueError if it is absent or empty.
    :param state: Pipeline state object (ImportGraphState)
    :return: Validated list of text chunk dicts
    :raises ValueError: If chunks is not a non-empty list
    """
    texts_to_embed = state.get("chunks")
    if not isinstance(texts_to_embed, list) or not texts_to_embed:
        logger.error(
            "Embedding input validation failed: chunks field is empty or not a valid list"
        )
        raise ValueError("No valid text chunks available for embedding")

    logger.info(
        f"Step 1: Input validation passed — {len(texts_to_embed)} chunk(s) to process"
    )
    return texts_to_embed


def step_2_init_model():
    """
    Step 2: Initialise the BGE-M3 model instance (singleton).
    Calls get_bge_m3_ef() to ensure the model is loaded only once globally.
    :return: Valid BGE-M3 embedding function instance
    :raises ValueError: If the model fails to load (wrong path, OOM, missing dependency)
    """
    try:
        ef = get_bge_m3_ef()
        if ef is None:
            raise ValueError(
                "BGE-M3 model instance is None: pymilvus.model module not found or model failed to load"
            )
        logger.info("Step 2: BGE-M3 model initialised successfully (singleton)")
        return ef
    except Exception as e:
        error_msg = f"BGE-M3 model initialisation failed: {e} — check model path and environment variable configuration"
        logger.error(error_msg)
        raise ValueError(error_msg)


def step_3_generate_embeddings(
    texts_to_embed: List[Dict[str, Any]], bge_m3_ef: Any
) -> List[Dict[str, Any]]:
    """
    Step 3: Batch-generate dense + sparse dual vectors for all chunks.
    Each batch is processed independently; a failed batch retains the original chunk data
    without interrupting the overall process.
    Input text format: "Product: <item_name>, Description: <content>" when item_name is present,
    otherwise just <content>. Prepending item_name keeps it within the first ~128 tokens
    where BERT-based models concentrate most attention.
    :param texts_to_embed: Validated chunk list; each dict must have item_name and content keys
    :param bge_m3_ef: BGE-M3 model instance from step 2
    :return: Chunk list with dense_vector and sparse_vector fields added; failed batches keep original data
    """
    output_data = []
    # Tune batch_size based on available VRAM: larger VRAM → larger batch
    batch_size = 5

    total = len(texts_to_embed)
    for i in range(0, total, batch_size):
        batch_texts = texts_to_embed[i : i + batch_size]
        start_idx, end_idx = i + 1, min(i + len(batch_texts), total)

        try:
            input_texts = []
            for doc in batch_texts:
                item_name = doc["item_name"]
                content = doc["content"]
                text = (
                    f"Product: {item_name}, Description: {content}"
                    if item_name
                    else content
                )
                input_texts.append(text)

            # Returns {"dense": [dense_vectors], "sparse": [sparse_vectors]}
            docs_embeddings = generate_embeddings(input_texts)
            if not docs_embeddings:
                logger.warning(
                    f"Chunks {start_idx}-{end_idx}: embedding returned empty — retaining original data"
                )
                output_data.extend(batch_texts)
                continue

            for j, doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            logger.info(
                f"Chunks {start_idx}-{end_idx}: dual vectors generated successfully"
            )

        except Exception as e:
            logger.error(
                f"Chunks {start_idx}-{end_idx}: vector generation failed, retaining original data — {str(e)}",
                exc_info=True,
            )
            output_data.extend(batch_texts)
            continue

    return output_data


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    Node: BGE-M3 Text Embedding (node_bge_embedding)
    Pipeline: validate input → init model → batch generate dual vectors → update state
    :param state: Pipeline state object; must contain chunks and task_id
    :return: Updated state with dense_vector and sparse_vector added to each chunk
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> Starting node: [BGE-M3 Embedding] {current_node}")

    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 text embedding started ---")

    try:
        # Step 1: Validate input chunks
        texts_to_embed = step_1_validate_input(state)

        # Step 2: Initialise BGE-M3 model (singleton — loaded once per process)
        bge_m3_ef = step_2_init_model()

        # Step 3: Batch-generate dual vectors and bind them to each chunk
        output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)

        # Step 4: Write vectorised chunks back to state for downstream nodes
        state['chunks'] = output_data
        logger.info(
            f"--- BGE-M3 embedding complete — {len(output_data)} chunk(s) processed ---"
        )
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        logger.error(f"BGE-M3 embedding node failed: {str(e)}", exc_info=True)

    return state


if __name__ == '__main__':
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    test_state = ImportGraphState(
        {
            "task_id": "test_task_embedding_001",
            "chunks": [
                {
                    "content": "This is the content of the first test document, used to verify embedding works correctly.",
                    "title": "Test Document Title",
                    "item_name": "Test Item",
                    "file_title": "test_file.pdf",
                },
                {
                    "content": "This is the content of the second test document, used to verify batch processing logic.",
                    "title": "Test Document Title 2",
                    "item_name": "Test Item",
                    "file_title": "test_file.pdf",
                },
            ],
        }
    )

    logger.info("=== BGE-M3 embedding node local test started ===")
    try:
        result_state = node_bge_embedding(test_state)
        result_chunks = result_state.get("chunks", [])

        logger.info("=== BGE-M3 embedding node local test complete ===")
        logger.info(f"Task ID: {test_state.get('task_id')}")
        logger.info(f"Expected chunks: 2 | Processed chunks: {len(result_chunks)}")

        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"Chunk {idx + 1}: dense vector {'OK' if has_dense else 'MISSING'} | sparse vector {'OK' if has_sparse else 'MISSING'}"
            )
            # logger.info(f"Chunk {idx + 1} data: {chunk}")

    except Exception as e:
        logger.error(
            f"=== BGE-M3 embedding node local test failed === {str(e)}", exc_info=True
        )
        logger.warning(
            "Troubleshooting: check BGE-M3 model path, available VRAM, and environment variable configuration"
        )
