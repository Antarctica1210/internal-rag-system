import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def _as_entity_list(state_list) -> List[Dict[str, Any]]:
    """
    Normalise upstream node output into a flat list of entity dicts.
    Handles:
    - Milvus Hit objects: {"entity": {...}, "distance": ...}
    - Plain dicts: nested {"entity": {...}} or flat {key: value}
    - Objects with a .get() method
    """
    out: List[Dict[str, Any]] = []
    for doc in state_list or []:
        if not doc:
            continue

        final_ent = {}

        # --- case A: Milvus Hit object (has .entity and .id attributes) ---
        if hasattr(doc, "entity") and hasattr(doc, "id"):
            entity_content = doc.entity
            if hasattr(entity_content, "to_dict"):
                final_ent = entity_content.to_dict()
            elif isinstance(entity_content, dict):
                final_ent = entity_content.copy()
            else:
                # fallback for different SDK versions
                try:
                    final_ent = dict(entity_content)
                except:
                    pass

            # use outer id when chunk_id is absent
            if "id" not in final_ent and "chunk_id" not in final_ent:
                final_ent["id"] = doc.id

            if hasattr(doc, "distance"):
                final_ent["score"] = doc.distance

        # --- case B: plain dict (mock data or pre-formatted) ---
        elif isinstance(doc, dict):
            if "entity" in doc:
                # nested structure: {entity: {...}, id: ..., distance: ...}
                ent = doc["entity"]
                if isinstance(ent, dict):
                    final_ent = ent.copy()
                if "id" in doc and "id" not in final_ent:
                    final_ent["id"] = doc["id"]
                if "distance" in doc:
                    final_ent["score"] = doc["distance"]
            else:
                # flat dict — use as-is
                final_ent = doc

        # --- case C: object with a .get() method ---
        elif hasattr(doc, "get"):
            ent = doc.get("entity") or doc
            if isinstance(ent, dict):
                final_ent = ent

        if final_ent and isinstance(final_ent, dict):
            out.append(final_ent)

    return out


def reciprocal_rank_fusion(
    source_weights: list,
    k: int = 60,
    max_results: int = None,
) -> List[tuple]:
    """
    Weighted Reciprocal Rank Fusion (RRF).
    :param source_weights: list of (doc_list, weight) tuples,
                           e.g. [([doc1, doc2], 1.0), ([doc2, doc3], 0.8)]
    :param k: RRF smoothing constant (default 60); dampens the advantage of top-ranked docs
    :param max_results: return only the top-N results; None returns all
    :return: [(doc, rrf_score), ...] sorted by score descending
    """
    score_map = {}
    chunk_map = {}

    for docs, weight in source_weights:
        for rank, item in enumerate(docs, start=1):
            chunk_id = item.get("chunk_id") or item.get("id")

            if not chunk_id:
                logger.warning(
                    f"RRF: item missing chunk_id/id: {list(item.keys()) if isinstance(item, dict) else item}"
                )
                continue

            # RRF formula: score += weight * (1 / (k + rank))
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (
                1.0 / (k + rank)
            )

            # keep only the first occurrence of each document
            chunk_map.setdefault(chunk_id, item)

    merged = [
        (chunk_map[chunk_id], score)
        for chunk_id, score in score_map.items()
    ]
    merged.sort(key=lambda x: x[1], reverse=True)

    if max_results is not None:
        merged = merged[:max_results]

    return merged


def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) node.
    Merges results from multiple retrieval sources (embedding search, HyDE search, etc.)
    using RRF — a training-free algorithm that scores documents solely by their rank
    positions across different result lists.

    Steps:
    1. Extract embedding_chunks and hyde_embedding_chunks from state.
    2. Normalise all chunks into a uniform entity dict format.
    3. Assign source weights and run RRF fusion.
    4. Truncate to top-K and return rrf_chunks.
    """
    logger.info("--- node_rrf start ---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))

    logger.info(
        f"RRF inputs: embedding={len(embedding_chunks)}, hyde={len(hyde_embedding_chunks)}"
    )

    if embedding_chunks:
        logger.debug(
            f"embedding chunk_ids (first 5): {[c.get('chunk_id') for c in embedding_chunks[:5]]}"
        )
    if hyde_embedding_chunks:
        logger.debug(
            f"hyde chunk_ids (first 5): {[c.get('chunk_id') for c in hyde_embedding_chunks[:5]]}"
        )

    source_weights = [(embedding_chunks, 1.0), (hyde_embedding_chunks, 1.0)]

    rrf_res = reciprocal_rank_fusion(source_weights, k=60, max_results=10)
    rrf_chunks = [doc for doc, score in rrf_res]

    logger.info(f"RRF output: {len(rrf_chunks)} chunk(s) after fusion")

    add_done_task(
        state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    return {"rrf_chunks": rrf_chunks}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> starting node_rrf test")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "How do I operate the HAK 180 hot stamping machine?",
        "rewritten_query": "What are the specific operating steps for the HAK 180 hot stamping machine?",
        "item_names": ["HAK 180 hot stamping machine"],
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> test result summary:")
        print(f"input counts: embedding={emb_cnt}, hyde={hyde_cnt}")
        print(f"output count: {len(rrf_chunks)}")
        print("-" * 30)
        print("final ranking:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"rank {i}: id={doc_id}, content={content}...")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"unhandled exception during test: {e}")
