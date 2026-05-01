import sys
import os
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    hybrid_search,
    get_milvus_client,
)
from app.core.logger import logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def node_search_embedding(state):
    """
    Core node: hybrid Milvus vector search based on the confirmed product name(s) and
    rewritten user query.
    Pipeline: embed query → build filtered hybrid search request → run dense+sparse search → return results.
    :param state: graph state dict with keys:
                  {
                      "session_id": str,
                      "rewritten_query": str,   # self-contained rewritten question (includes product name)
                      "item_names": list[str],  # confirmed normalised product names
                      "is_stream": bool | None
                  }
    :return: {"embedding_chunks": List[Dict]}  # Milvus hits; empty list when nothing found
    """
    logger.info("--- node_search_embedding start ---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state["is_stream"]
    )

    query = state.get("rewritten_query")
    item_names = state.get("item_names")

    logger.info(f"inputs: query='{query}', item_names={item_names}")

    # embed the rewritten query — BGE-M3 dense + sparse vectors
    logger.info(
        f"generating embeddings for: {query[:50]}..."
        if len(query) > 50
        else f"generating embeddings for: '{query}'"
    )
    embeddings = generate_embeddings([query])

    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]
    logger.debug(
        f"embeddings ready: dense_dim={len(dense_vec)}, sparse_len={len(sparse_vec)}"
    )

    collection_name = os.environ.get("CHUNKS_COLLECTION")
    logger.info(f"target collection: '{collection_name}'")

    # skip retrieval when no product names are confirmed
    if not item_names:
        logger.warning("item_names is empty, skipping retrieval")
        return {"embedding_chunks": []}

    # build Milvus filter expression: item_name in ["Product A", "Product B"]
    quoted = ", ".join(f'"{v}"' for v in item_names)
    expr = f"item_name in [{quoted}]"
    logger.info(f"filter expression: {expr}")

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=10,  # fetch more candidates; top-5 returned after reranking
    )

    logger.info("executing Milvus hybrid search...")
    client = get_milvus_client()
    res = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),  # dense / sparse weight ratio
        norm_score=True,            # normalise scores to 0-1
        limit=5,
        output_fields=["chunk_id", "content", "item_name"],
    )

    hit_count = len(res[0]) if res and len(res) > 0 else 0
    logger.info(f"node_search_embedding done — {hit_count} chunk(s) retrieved")
    if hit_count > 0:
        logger.debug(f"top-1 result: {res[0][0]}")

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    return {"embedding_chunks": res[0] if res else []}


if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 hot stamping machine user manual",
        "item_names": ["HAK 180 hot stamping machine"],
        "is_stream": False,
    }

    print("\n>>> starting node_search_embedding test...")
    try:
        result = node_search_embedding(test_state)
        logger.info(f"retrieval result: {result}")
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> test complete — {len(chunks)} chunk(s) retrieved")

        if chunks:
            print("\n>>> top-1 result:")
            top1 = chunks[0]
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> warning: no results — check Milvus data or item_names filter")

    except Exception as e:
        logger.error(f"test failed: {e}", exc_info=True)
