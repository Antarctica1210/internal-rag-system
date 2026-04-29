import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    Step 1: Use an LLM to generate a hypothetical document for the user query.
    HyDE core idea: generate a "fictional but relevant" document with the LLM, then use
    that document's vector for retrieval — bridging the semantic gap between a short query
    and long documents.
    :param rewritten_query: rewritten user query
    :return: LLM-generated hypothetical document text
    """
    if not rewritten_query:
        logger.error("Step 1 error: rewritten_query is empty")
        raise ValueError("rewritten_query cannot be empty")

    logger.info(f"Step 1: generating hypothetical document (HyDE), query: {rewritten_query}")

    try:
        llm = get_llm_client()
        hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        logger.debug(f"Step 1: prompt loaded, length: {len(hyde_prompt)}")

        response = llm.invoke(hyde_prompt)
        hyde_doc = response.content

        logger.info(f"Step 1: hypothetical document generated, length: {len(hyde_doc)} chars")
        logger.debug(f"Step 1: document preview: {hyde_doc[:50]}...")

        return hyde_doc

    except Exception as e:
        logger.error(f"Step 1: failed to generate hypothetical document: {e}")
        raise e


def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),  # weighted towards dense vectors
    norm_score: bool = True,
    output_fields=["chunk_id", "content", "item_name"],
):
    """
    Step 2: Embed the combined "rewritten query + hypothetical document" and run a
    hybrid Milvus search to retrieve matching chunks.
    :param rewritten_query: rewritten user query
    :param hyde_doc: hypothetical document from step 1
    :param item_names: product name list for metadata filtering (item_name in [...])
    :param req_limit: candidate recall count for Milvus search
    :param top_k: final number of top-K results to return
    :param ranker_weights: hybrid search weights (dense, sparse)
    :param norm_score: whether to normalise scores
    :param output_fields: fields to include in results
    :return: search result list
    """
    if not rewritten_query:
        raise ValueError("rewritten_query cannot be empty")
    if not hyde_doc:
        raise ValueError("hypothetical_doc cannot be empty")

    # concatenate query and hypothetical doc for richer semantic context
    combined_text = rewritten_query + " " + hyde_doc
    logger.info(f"Step 2: combined query + HyDE doc, total length: {len(combined_text)}")

    logger.info("Step 2: generating hybrid embeddings...")
    embeddings = generate_embeddings([combined_text])

    collection_name = os.environ.get("CHUNKS_COLLECTION")
    if not collection_name:
        logger.error("Step 2 error: env var CHUNKS_COLLECTION is not set")
        return []

    logger.info(f"Step 2: preparing hybrid search in collection '{collection_name}'")

    expr = None
    if item_names:
        quoted = ", ".join(f'"{v}"' for v in item_names)
        expr = f"item_name in [{quoted}]"
        logger.info(f"Step 2: applying filter: {expr}")
    else:
        logger.info("Step 2: no product name filter — searching entire collection")

    try:
        reqs = create_hybrid_search_requests(
            dense_vector=embeddings.get("dense")[0],
            sparse_vector=embeddings.get("sparse")[0],
            expr=expr,
            limit=req_limit,
        )

        client = get_milvus_client()
        if not client:
            logger.error("Step 2 error: unable to connect to Milvus")
            return []

        logger.info(f"Step 2: running hybrid search — weights={ranker_weights}, top_k={top_k}")
        res = hybrid_search(
            client=client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=top_k,
            output_fields=list(output_fields),
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: search complete — {hit_count} matching chunk(s) found")

        return res

    except Exception as e:
        logger.error(f"Step 2: exception during retrieval: {e}")
        return []


def node_search_embedding_hyde(state):
    """
    HyDE (Hypothetical Document Embedding) retrieval node.
    Core idea: generate a hypothetical answer with the LLM, embed it, and use it for
    retrieval — improving recall for short or ambiguous queries.

    Steps:
    1. Extract rewritten_query and confirmed item_names from state.
    2. Generate a hypothetical document via LLM (step 1).
    3. Concatenate query + hypothetical doc, embed with BGE-M3, run hybrid Milvus search (step 2).
    4. Return retrieved chunks and the hypothetical document.

    :param state: graph state dict with session_id, rewritten_query, item_names, etc.
    :return: {"hyde_embedding_chunks": List[Dict], "hyde_doc": str}
    """
    logger.info("--- node_search_embedding_hyde start ---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # fall back to original_query if rewritten_query is absent
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        rewritten_query = state.get("original_query")

    if not rewritten_query:
        logger.error("HyDE node error: no valid query found (rewritten_query and original_query are both empty)")
        return {}

    item_names = state.get("item_names")
    logger.info(f"HyDE inputs: query='{rewritten_query}', item_names={item_names}")

    # step 1: generate hypothetical document
    hyde_doc = ""
    try:
        logger.info("Step 1: generating hypothetical document (HyDE doc)...")
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
        logger.info(f"Step 1: hypothetical document ready (length: {len(hyde_doc)})")
        logger.debug(f"doc preview: {hyde_doc[:100]}...")
    except Exception as e:
        logger.error(f"Step 1 (generate hypothetical doc) exception: {e}", exc_info=True)
        return {}

    # step 2: retrieve chunks using query + hypothetical doc
    try:
        logger.info("Step 2: running Milvus hybrid search with hypothetical document...")
        res = step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            top_k=5,
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: search done — {hit_count} chunk(s) retrieved")

        if hit_count > 0:
            first_hit = res[0][0]
            score = first_hit.get("distance")
            content_preview = first_hit.get("entity", {}).get("content", "")[:30]
            logger.debug(f"top-1: score={score}, content='{content_preview}...'")

        return {
            "hyde_embedding_chunks": res[0] if res else [],
            "hyde_doc": hyde_doc,
        }
    except Exception as e:
        logger.error(f"Step 2 (embed + search) exception: {e}", exc_info=True)
        return {}
    finally:
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        logger.info("--- node_search_embedding_hyde end ---")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> starting node_search_embedding_hyde test")
    print("=" * 50)

    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "How do I operate the HAK 180 hot stamping machine?",
        "rewritten_query": "What are the specific operating steps for the HAK 180 hot stamping machine?",
        "item_names": ["HAK 180 hot stamping machine"],
        "is_stream": False,
    }

    try:
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> test result summary:")
        print(f"HyDE doc generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"doc preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"chunks found: {len(chunks)}, content: {chunks}")
        if chunks:
            print(f"top chunk score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"unhandled exception during test: {e}")
