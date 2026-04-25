import os
import sys
from typing import List, Dict, Any

from pymilvus import DataType

from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task
from app.utils.escape_milvus_string_utils import escape_milvus_string

CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def step_1_check_input(state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    """
    Step 1: Validate input data before inserting into Milvus.
    Checks that chunks is a non-empty list, that the first chunk contains dense_vector,
    and extracts the vector dimension for collection creation.
    :param state: Pipeline state dict containing chunks produced by the upstream embedding node
    :return: Tuple of (validated chunk list, dense vector dimension)
    :raises ValueError: If any validation check fails
    """
    chunks_json_data = state.get("chunks")
    if not chunks_json_data:
        logger.error("Milvus import validation failed: chunks field is empty")
        raise ValueError("chunks is empty — cannot proceed with Milvus import")
    if not isinstance(chunks_json_data, list) or len(chunks_json_data) == 0:
        logger.error("Milvus import validation failed: chunks is not a non-empty list")
        raise ValueError("chunks must be a non-empty list")
    first_chunk = chunks_json_data[0]
    if 'dense_vector' not in first_chunk:
        logger.error(
            "Milvus import validation failed: dense_vector field missing — upstream embedding node may have failed"
        )
        raise ValueError(
            "dense_vector field is missing — check that the upstream embedding node ran successfully"
        )

    vector_dimension = len(first_chunk['dense_vector'])
    item_name = first_chunk.get('item_name', 'Unknown')
    logger.info(
        f"Step 1: Validation passed — chunks: {len(chunks_json_data)}, vector dim: {vector_dimension}, item name: {item_name}"
    )

    return chunks_json_data, vector_dimension


def create_collection(client, collection_name: str, vector_dimension: int):
    """
    Create a Milvus collection with schema and indexes.
    Schema includes business fields, dual vector fields, and an auto-increment primary key.
    :param client: Connected MilvusClient instance
    :param collection_name: Name of the collection to create
    :param vector_dimension: Dense vector dimension — must match the embedding model output
                             (BGE-M3 = 1024, BGE-base = 768, BGE-small = 384)
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)

    schema.add_field(
        field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True
    )
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    # item_name is also used as the dedup key for idempotent deletes
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension
    )

    index_params = client.prepare_index_params()
    # HNSW + COSINE for dense vector — best recall/speed trade-off for BGE-M3
    # Recommended M/efConstruction by dataset size:
    #   ~10k rows:  M=16, efConstruction=200
    #   ~50k rows:  M=32, efConstruction=300
    #   ~100k rows: M=64, efConstruction=400
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # SPARSE_INVERTED_INDEX + IP for sparse vector — standard for keyword-weight vectors
    # DAAT_MAXSCORE skips zero-value dimensions; quantization=none preserves float precision
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"},
    )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    logger.info(f"Collection created: {collection_name}, vector dim: {vector_dimension}")


def step_2_prepare_collection(vector_dimension: int):
    """
    Step 2: Connect to Milvus and prepare the target collection.
    Creates the collection with schema and indexes if it does not already exist.
    :param vector_dimension: Dense vector dimension extracted in step 1
    :return: Connected MilvusClient instance with the collection ready
    :raises ValueError: If the client cannot be obtained or collection name is not configured
    """
    logger.info(f"Step 2: Preparing Milvus environment — target collection: {CHUNKS_COLLECTION_NAME}")
    client = get_milvus_client()
    if client is None:
        logger.error("Failed to obtain Milvus client: get_milvus_client() returned None")
        raise ValueError("Milvus connection failed: get_milvus_client() returned None")
    if not CHUNKS_COLLECTION_NAME:
        logger.error("CHUNKS_COLLECTION_NAME is not configured")
        raise ValueError("CHUNKS_COLLECTION collection name is not configured")

    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        logger.info(f"Collection [{CHUNKS_COLLECTION_NAME}] not found — creating schema and indexes")
        create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)
    else:
        logger.info(f"Collection [{CHUNKS_COLLECTION_NAME}] already exists — reusing")

    return client


def _clear_chunks_by_item_name(client, collection_name: str, item_name: str):
    """
    Delete all chunks matching the given item_name from the collection.
    Used for idempotent writes: removes stale data before inserting a fresh batch.
    :param client: MilvusClient instance
    :param collection_name: Target collection name
    :param item_name: Item name whose existing records should be removed
    :raises ValueError: If the delete operation fails (propagated to abort the import)
    """
    i_name = (item_name or "").strip()
    if not i_name:
        logger.warning("Idempotent delete skipped: item_name is empty")
        return
    if not collection_name:
        logger.warning("Idempotent delete skipped: collection name is not configured")
        return

    try:
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Idempotent delete skipped: collection [{collection_name}] does not exist")
            return

        safe_item_name = escape_milvus_string(i_name)
        filter_expr = f'item_name == "{safe_item_name}"'
        logger.info(f"Idempotent delete: removing existing records for item_name={i_name} from [{collection_name}]")

        client.delete(collection_name=collection_name, filter=filter_expr)

        # Flush to ensure the delete is applied before the subsequent insert
        if hasattr(client, "flush"):
            try:
                client.flush(collection_name=collection_name)
            except Exception as e:
                logger.warning(f"Flush after delete failed (non-fatal): {str(e)}")

        logger.info(f"Idempotent delete complete: removed records for item_name={i_name}")
    except Exception as e:
        logger.error(f"Idempotent delete failed: item_name={i_name} — {str(e)}", exc_info=True)
        raise ValueError(f"Idempotent delete failed (item_name={i_name}): {e}")


def step_3_clean_old_data(client, chunks_json_data: List[Dict[str, Any]]):
    """
    Step 3: Idempotent cleanup — delete existing records by item_name before inserting.
    Deduplicates item_names across chunks to avoid redundant delete operations.
    Supports batches that span multiple item_names (each is cleaned individually).
    :param client: MilvusClient instance
    :param chunks_json_data: Chunk list to be inserted
    """
    # Extract unique non-empty item_names using a walrus-operator set comprehension
    item_names = sorted(
        {
            name
            for x in chunks_json_data or []
            if (name := str(x.get("item_name", "")).strip())
        }
    )

    if not item_names:
        logger.warning("Idempotent cleanup skipped: no valid item_name found in chunks")
        return
    if len(item_names) > 1:
        logger.warning(f"Multiple item_names detected — cleaning each: {item_names}")

    for i_name in item_names:
        _clear_chunks_by_item_name(client, CHUNKS_COLLECTION_NAME, i_name)


def step_4_insert_data(
    client, chunks_json_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Step 4: Batch-insert chunks into Milvus and write back auto-generated chunk_ids.
    Removes any manually set chunk_id before insert (collection uses auto_id=True).
    After insert, backfills Milvus-generated IDs onto the original chunk dicts
    so downstream nodes can reference them.
    :param client: MilvusClient instance
    :param chunks_json_data: Chunk list to insert
    :return: Chunk list with chunk_id backfilled from Milvus
    """
    data_to_insert = []
    for item in chunks_json_data:
        item_copy = item.copy()
        if isinstance(item_copy, dict) and "chunk_id" in item_copy:
            item_copy.pop("chunk_id", None)
        data_to_insert.append(item_copy)

    logger.info(f"Step 4: Inserting {len(data_to_insert)} chunk(s) into Milvus")
    insert_result = client.insert(
        collection_name=CHUNKS_COLLECTION_NAME, data=data_to_insert
    )
    insert_count = insert_result.get('insert_count', 0)
    logger.info(f"Insert complete: {insert_count} record(s) written — result: {insert_result}")

    inserted_ids = insert_result.get('ids', [])
    if inserted_ids and len(inserted_ids) == len(chunks_json_data):
        logger.info(f"Backfilling {len(inserted_ids)} auto-generated chunk_id(s) onto chunks")
        for idx, item in enumerate(chunks_json_data):
            item['chunk_id'] = str(inserted_ids[idx])
        logger.info("chunk_id backfill complete")
    else:
        logger.warning(
            f"chunk_id backfill skipped: ID count ({len(inserted_ids)}) does not match chunk count ({len(chunks_json_data)})"
        )

    return chunks_json_data


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Milvus Chunk Import (node_import_milvus)
    Pipeline: validate input → prepare collection → idempotent cleanup → batch insert → update state
    :param state: Pipeline state dict; must contain chunks (with vectors) and task_id
    :return: Updated state with chunk_id backfilled on each chunk
    :raises ValueError: On any step failure to prevent dirty writes
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> Starting node: [Milvus Import] {current_node}")
    add_running_task(state["task_id"], current_node)
    logger.info("--- Milvus chunk import started ---")

    try:
        # Step 1: Validate chunks and extract vector dimension
        chunks_json_data, vector_dimension = step_1_check_input(state)
        # Step 2: Connect to Milvus and create collection if absent
        client = step_2_prepare_collection(vector_dimension)
        # Step 3: Delete stale records by item_name (idempotent)
        step_3_clean_old_data(client, chunks_json_data)
        # Step 4: Batch insert and backfill auto-generated chunk_ids
        updated_chunks = step_4_insert_data(client, chunks_json_data)
        # Step 5: Write updated chunks back to state for downstream nodes
        state["chunks"] = updated_chunks

        logger.info("--- Milvus chunk import complete ---")
    except Exception as e:
        logger.error(f"Milvus import node failed: {str(e)}", exc_info=True)
        raise ValueError(f"Milvus import error: {e}")

    return state


if __name__ == '__main__':
    import sys
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus test text 1",
                "title": "Test Title",
                "item_name": "Test_Item_Milvus",
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            }
        ],
    }

    print("Running Milvus import node test...")
    try:
        if not os.getenv("MILVUS_URL"):
            print("MILVUS_URL is not set — cannot connect to Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("CHUNKS_COLLECTION is not set")
        else:
            result_state = node_import_milvus(test_state)

            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"Milvus import test passed — generated chunk_id: {chunks[0]['chunk_id']}")
            else:
                print("Test failed: chunk_id was not returned")

    except Exception as e:
        print(f"Test failed: {e}")
