import os
import sys
from typing import List, Dict, Any, Tuple

from pymilvus import MilvusClient, DataType
from langchain_core.messages import SystemMessage, HumanMessage

from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.normalize_sparse_vector import normalize_sparse_vector
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- Configuration ---
# Number of chunks fed to the LLM for item name recognition; top-k to stay within context limits
DEFAULT_ITEM_NAME_CHUNK_K = 5
# Max characters per chunk to prevent a single chunk from consuming the entire LLM context budget
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# Total context character limit; sized for mainstream LLM input constraints
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """
    Step 1: Extract and validate pipeline inputs for item name recognition.
    Extracts the file title and chunk list from state with multi-level fallbacks
    to prevent downstream failures caused by missing values.
    State keys consumed (produced by upstream nodes):
        - state["file_title"]: primary file title
        - state["file_name"]: fallback if file_title is absent
        - state["chunks"]: list of text chunk dicts (each has title/content fields)
    :return: Tuple of (resolved file title, validated chunk list)
    """
    # Prefer file_title; fall back to file_name, then empty string
    file_title = state.get("file_title", "") or state.get("file_name", "")
    chunks = state.get("chunks") or []

    # Secondary fallback: extract file_title from the first valid chunk
    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning("No valid file_title in state — extracted fallback title from first chunk")

    if not file_title:
        logger.warning("state is missing both file_title and file_name — LLM recognition accuracy may be reduced")

    if not isinstance(chunks, list) or not chunks:
        logger.warning("state chunks is empty or not a list — skipping item name recognition")
        return file_title, []

    logger.info(f"Step 1: Input validation complete — {len(chunks)} valid chunk(s) loaded")
    return file_title, chunks


def step_2_build_context(
    chunks: List[Dict],
    k: int = DEFAULT_ITEM_NAME_CHUNK_K,
    max_chars: int = CONTEXT_TOTAL_MAX_CHARS,
) -> str:
    """
    Step 2: Build a structured context string for LLM-based item name recognition.
    Limits the number of chunks to k, truncates oversized individual chunks,
    and caps the total context at max_chars to respect LLM input limits.
    :param chunks: List of chunk dicts; each must have "title" and "content" keys
    :param k: Maximum number of chunks to include; default 5
    :param max_chars: Total character limit for the assembled context; default 2500
    :return: Formatted context string ready to pass to the LLM; empty string if no valid chunks
    """
    if not chunks:
        return ""

    parts: List[str] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks[:k]):
        if not isinstance(chunk, dict):
            logger.debug(f"Chunk {idx + 1} is not a dict — skipped")
            continue

        chunk_title = chunk.get("title", "").strip()
        chunk_content = chunk.get("content", "").strip()

        if not (chunk_title or chunk_content):
            logger.debug(f"Chunk {idx + 1} has no content — skipped")
            continue

        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
            logger.debug(f"Chunk {idx + 1} content truncated to {SINGLE_CHUNK_CONTENT_MAX_LEN} characters")

        piece = f"[Chunk {idx + 1}]\nTitle: {chunk_title}\nContent: {chunk_content}"
        parts.append(piece)
        total_chars += len(piece)

        if total_chars > max_chars:
            logger.info(f"Context approaching character limit ({max_chars}) — stopped appending further chunks")
            break

    context = "\n\n".join(parts).strip()
    final_context = context[:max_chars]
    logger.info(f"Step 2: Context built — final length {len(final_context)} character(s)")
    return final_context


def step_3_call_llm(file_title: str, context: str) -> str:
    """
    Step 3: Call the LLM to identify the product name and model from the context.
    Falls back to file_title if the context is empty, the LLM returns nothing, or an error occurs.
    :param file_title: Resolved file title used as the fallback value
    :param context: Structured chunk context built in step 2
    :return: Cleaned product name string; falls back to file_title on failure
    """
    logger.info("Step 3: Calling LLM for item name recognition")

    if not context:
        logger.warning("Context is empty — skipping LLM call, using file title as item name")
        return file_title

    try:
        human_prompt = load_prompt(
            "item_name_recognition", file_title=file_title, context=context
        )
        system_prompt = load_prompt("product_recognition_system")
        logger.debug(
            f"Prompts ready — system: {len(system_prompt)} chars, human: {len(human_prompt)} chars"
        )

        llm = get_llm_client(json_mode=False)
        if not llm:
            logger.error("Failed to obtain LLM client — falling back to file title")
            return file_title

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        resp = llm.invoke(messages)

        item_name = getattr(resp, "content", "").strip()
        item_name = (
            item_name.replace(" ", "")
            .replace("\n", "")
            .replace("\r", "")
            .replace("\t", "")
        )

        if not item_name:
            logger.warning("LLM returned empty content — falling back to file title")
            return file_title

        logger.info(f"Step 3: LLM recognition successful — result: {item_name}")
        return item_name

    except Exception as e:
        logger.error(f"Step 3: LLM call failed — {str(e)}", exc_info=True)
        return file_title


def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], item_name: str):
    """
    Step 4: Write the recognised item name back to state and all chunk dicts.
    Ensures all chunks carry the same item_name for consistent downstream
    vector ingestion and retrieval.
    :param state: Pipeline state object (ImportGraphState)
    :param chunks: Validated chunk list from step 1
    :param item_name: Cleaned item name from step 3
    """
    state["item_name"] = item_name
    for chunk in chunks:
        chunk["item_name"] = item_name
    state["chunks"] = chunks
    logger.info(
        f"Step 4: item_name backfill complete — {len(chunks)} chunk(s) updated with item_name: {item_name}"
    )


def step_5_generate_vectors(item_name: str) -> Tuple[Any, Any]:
    """
    Step 5: Generate BGE-M3 dense + sparse dual vectors for the item name.
    - Dense vector: fixed 1024-dim float list capturing deep semantic meaning
    - Sparse vector: variable-length key-value dict capturing keyword/positional features
    :param item_name: Recognised item name from step 3
    :return: Tuple (dense_vector, sparse_vector); (None, None) if empty or on error
    """
    logger.info(f"Step 5: Generating BGE-M3 dual vectors for item name [{item_name}]")

    if not item_name:
        logger.warning("Item name is empty — skipping vector generation")
        return None, None

    try:
        vector_result = generate_embeddings([item_name])

        if vector_result and "dense" in vector_result and "sparse" in vector_result:
            dense_vector = vector_result["dense"][0]
            sparse_vector = vector_result["sparse"][0]
            logger.info("Step 5: BGE-M3 dense + sparse vectors generated successfully")
        else:
            logger.warning("Step 5: Vector generation returned empty result — cannot extract dual vectors")
            dense_vector, sparse_vector = None, None

    except Exception as e:
        logger.error(f"Step 5: Vector generation failed — {str(e)}", exc_info=True)
        dense_vector, sparse_vector = None, None

    return dense_vector, sparse_vector


def step_6_save_to_milvus(
    state: ImportGraphState,
    file_title: str,
    item_name: str,
    dense_vector,
    sparse_vector,
):
    """
    Step 6: Persist the item name, file title, and dual vectors to Milvus.
    Flow: validate config → get client → create collection if absent →
          idempotent delete → insert → load collection.
    :param state: Pipeline state object for final state sync
    :param file_title: Resolved file title
    :param item_name: Recognised item name (used as dedup key)
    :param dense_vector: 1024-dim float list from step 5
    :param sparse_vector: Sparse dict from step 5
    """
    milvus_uri = os.environ.get("MILVUS_URL")
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")

    if not all([milvus_uri, collection_name]):
        logger.warning(
            "Milvus config missing (MILVUS_URL / ITEM_NAME_COLLECTION) — skipping save"
        )
        return

    logger.info(
        f"Step 6: Saving item name [{item_name}] to Milvus collection [{collection_name}]"
    )

    try:
        client = get_milvus_client()
        if not client:
            logger.error("Failed to obtain Milvus client — skipping save")
            return

        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Collection [{collection_name}] not found — creating schema and indexes")
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            # Auto-increment INT64 primary key
            schema.add_field(
                field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True
            )
            schema.add_field(
                field_name="file_title", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="item_name", datatype=DataType.VARCHAR, max_length=65535
            )
            # BGE-M3 dense vector: fixed 1024 dims
            schema.add_field(
                field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024
            )
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

            index_params = client.prepare_index_params()
            # HNSW + COSINE for dense vector — best recall/speed trade-off for BGE-M3
            # M=16, efConstruction=200 is recommended for datasets up to ~10k entries
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            # SPARSE_INVERTED_INDEX + IP for sparse vector — standard for keyword-weight vectors
            # DAAT_MAXSCORE skips zero-value dimensions for faster retrieval
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"},
            )

            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"Collection [{collection_name}] created with schema and vector indexes")

        # Idempotent delete: remove existing entry with the same item_name before inserting
        clean_item_name = (item_name or "").strip()
        if clean_item_name:
            client.load_collection(collection_name=collection_name)
            safe_item_name = escape_milvus_string(clean_item_name)
            filter_expr = f'item_name=="{safe_item_name}"'
            client.delete(collection_name=collection_name, filter=filter_expr)
            logger.info(f"Idempotent delete complete — removed existing entry for [{clean_item_name}]")

        data = {"file_title": file_title, "item_name": item_name}
        if dense_vector is not None:
            data["dense_vector"] = dense_vector
        if sparse_vector is not None:
            data["sparse_vector"] = sparse_vector

        client.insert(collection_name=collection_name, data=[data])
        # Force-load collection after insert so data is immediately queryable and visible in Attu
        client.load_collection(collection_name=collection_name)

        state["item_name"] = item_name
        logger.info(
            f"Step 6: Item name [{item_name}] saved to Milvus collection [{collection_name}] — fields: {list(data.keys())}"
        )

    except Exception as e:
        logger.error(f"Step 6: Failed to save to Milvus — {str(e)}", exc_info=True)


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Item Name Recognition (node_item_name_recognition)
    Pipeline: extract inputs → build context → LLM recognition → backfill chunks → generate vectors → save to Milvus
    :param state: Pipeline state dict (ImportGraphState); must contain chunks, file_title, task_id
    :return: Updated state with item_name added and each chunk annotated with item_name
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> Starting node: [Item Name Recognition] {node_name}")
    add_running_task(state.get("task_id", ""), node_name)

    try:
        # Step 1: Extract and validate inputs
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            logger.warning(f">>> Node skipped: {node_name} — no valid chunks in state")
            return state

        # Step 2: Build LLM context from top-k chunks
        context = step_2_build_context(chunks)

        # Step 3: Call LLM to extract item name from context and file title
        item_name = step_3_call_llm(file_title, context)

        # Step 4: Write item name back to state and all chunk dicts
        step_4_update_chunks(state, chunks, item_name)

        # Step 5: Generate BGE-M3 dense + sparse vectors for the item name
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)

        # Step 6: Persist item name and vectors to Milvus
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        logger.info(
            f">>> Node complete: [Item Name Recognition] {node_name} — result: {item_name}, saved to Milvus"
        )

    except Exception as e:
        logger.error(
            f">>> Node failed: [Item Name Recognition] {node_name} — {str(e)}",
            exc_info=True,
        )
        state["item_name"] = "Unknown"

    return state


def test_node_item_name_recognition():
    """
    Local integration test for node_item_name_recognition.
    Simulates LangGraph state input and runs the full node pipeline end-to-end.
    Prerequisites:
        1. .env configured (MILVUS_URL, ITEM_NAME_COLLECTION, LLM credentials)
        2. Milvus, LLM, and BGE-M3 services are accessible
        3. Prompt templates (item_name_recognition, product_recognition_system) exist
    """
    logger.info("=== Starting item name recognition node local test ===")
    try:
        mock_state = ImportGraphState(
            {
                "task_id": "test_task_123456",
                "file_title": "Huawei Mate60 Pro Smartphone User Manual",
                "file_name": "HuaweiMate60Pro_manual.pdf",
                "chunks": [
                    {
                        "title": "Product Overview",
                        "content": "The Huawei Mate60 Pro is a flagship smartphone released by Huawei in 2023, powered by the Kirin 9000S chip. It supports satellite calling, features a 6.82-inch display, and a resolution of 2700x1224.",
                    },
                    {
                        "title": "Camera",
                        "content": "The Huawei Mate60 Pro has a triple rear camera: 50MP main + 12MP ultra-wide + 48MP telephoto, with 5x optical zoom and 100x digital zoom.",
                    },
                    {
                        "title": "Battery",
                        "content": "5000mAh battery with 88W wired SuperCharge, 50W wireless SuperCharge, and reverse wireless charging.",
                    },
                ],
            }
        )

        result_state = node_item_name_recognition(mock_state)

        logger.info("=== Item name recognition node local test complete ===")
        logger.info(f"Task ID: {result_state.get('task_id')}")
        logger.info(f"Recognised item name: {result_state.get('item_name')}")
        logger.info(f"Chunk count: {len(result_state.get('chunks', []))}")
        logger.info(
            f"First chunk item_name: {result_state.get('chunks', [{}])[0].get('item_name')}"
        )

        # Optional: verify the saved entry in Milvus
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"],
            )
            logger.info(f"Milvus query result: {res}")

    except Exception as e:
        logger.error(f"Item name recognition node local test failed — {str(e)}", exc_info=True)


if __name__ == "__main__":
    test_node_item_name_recognition()
