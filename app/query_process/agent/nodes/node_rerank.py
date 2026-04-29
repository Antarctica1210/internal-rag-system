import sys
from app.utils.task_utils import *

from dotenv import load_dotenv
from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task
from app.core.logger import logger


load_dotenv()

RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 1
RERANK_GAP_RATIO: float = 0.25
RERANK_GAP_ABS: float = 0.5


def step_1_merge_docs(state):
    """
    Step 1: Document merging and normalisation.
    Combines heterogeneous results from multiple retrieval sources (local KB + web search)
    into a uniform format that the reranker model can process.

    Input sources:
    1. rrf_chunks (List[Dict]): local KB results after RRF fusion.
       Key fields: chunk_id, content, title/item_name.
    2. web_search_docs (List[Dict]): MCP web search results.
       Key fields: snippet, title, url.

    Output (List[Dict]), one entry per document:
      - text:             core text for reranking (content or snippet)
      - title:            document title
      - doc_id/chunk_id:  unique identifier (None for web results)
      - url:              source URL (empty for local results)
      - source:           "local" or "web"
    """
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []

    logger.info(
        f"Step 1: merging documents — local RRF: {len(rrf_docs)}, web: {len(web_docs)}"
    )
    doc_items = []

    # --- local KB documents (rrf_chunks) ---
    # upstream RRF node already ran _as_entity_list, so docs are plain dicts
    for i, doc in enumerate(rrf_docs):
        # defensive: unwrap nested "entity" dict if present
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc

        if not isinstance(entity, dict):
            logger.warning(f"unexpected local doc format (index={i}): {type(entity)}")
            continue

        content = entity.get("content")
        if not content:
            logger.debug(f"skipping empty local doc (index={i}, keys={list(entity.keys())})")
            continue

        doc_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name") or ""

        doc_items.append({
            "text": content,
            "doc_id": doc_id,
            "chunk_id": doc_id,  # kept for backward compatibility
            "title": title,
            "url": "",
            "source": "local",
        })

    # --- web search documents (web_search_docs) ---
    for i, doc in enumerate(web_docs):
        # prefer "snippet"; fall back to "content"
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()

        if not text:
            logger.debug(f"skipping empty web doc (index={i})")
            continue

        doc_items.append({
            "text": text,
            "doc_id": None,
            "chunk_id": None,
            "title": title,
            "url": url,
            "source": "web",
        })

    logger.info(f"Step 1: merge complete — {len(doc_items)} normalised doc(s)")
    return doc_items


def step_2_rerank_docs(state, doc_items):
    """
    Step 2: Rerank documents using the BGE reranker model.
    Input:  doc_items — list of normalised dicts from step 1
    Output: scored and sorted list of dicts with a "score" field

    sentence_pairs format: [[query, passage], ...] — order is strict (query first).
    Scores are relative; higher means more relevant. Sort descending to get the ranked list.
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""

    if not doc_items or not question:
        logger.warning("Step 2: skipping rerank (no documents or no question)")
        return []

    logger.info(f"Step 2: reranking {len(doc_items)} document(s)")

    texts = [x["text"] for x in doc_items]
    try:
        reranker = get_reranker_model()

        sentence_pairs = [[question, t] for t in texts]
        logger.info("Step 2: computing relevance scores...")
        scores = reranker.compute_score(sentence_pairs)

        scored_docs = []
        for item, text, score in zip(doc_items, texts, scores):
            scored_docs.append({
                "text": text,
                "score": float(score),
                "source": item.get("source") or "",
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "url": item.get("url") or "",
                "title": item.get("title") or "",
            })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs

    except Exception as e:
        logger.error(f"Step 2: reranking failed: {e}", exc_info=True)
        # degrade gracefully — return original order with score=0 so the pipeline continues
        return [
            {
                "text": x.get("text"),
                "score": 0.0,
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
            }
            for x in doc_items
        ]


def step_3_topk(scored_docs):
    """
    Step 3: Dynamic top-K truncation (up to RERANK_MAX_TOPK=10).
    Rather than always taking the top N, this detects a score "cliff" — a sudden drop
    between consecutive documents — and cuts off there to avoid including irrelevant results.

    Rules:
    - First RERANK_MIN_TOPK docs are always kept (no cliff check on them).
    - For subsequent positions: if the absolute gap >= RERANK_GAP_ABS OR the relative
      gap >= RERANK_GAP_RATIO, truncate at that position.
    - rel = (s1 - s2) / (|s1| + 1e-6)  — 1e-6 guards against division by zero.

    :param scored_docs: list of score-annotated dicts, already sorted descending by score
    :return: truncated list of top-K docs
    """
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK
    gap_ratio = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS

    topk = max_topk  # default: take the full allowed window
    if topk > min_topk:
        # scan from min_topk onward; the first min_topk docs are always retained
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")
            s2 = scored_docs[i + 1].get("score")

            gap = s1 - s2                        # always >= 0 (list is sorted descending)
            rel = gap / (abs(s1) + 1e-6)         # relative drop

            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(
                    f"Step 3: cliff detected @ index={i} "
                    f"(score {s1:.4f} → {s2:.4f}, gap={gap:.4f})"
                )
                topk = i + 1
                break

    topk_docs = scored_docs[:topk]
    logger.info(f"Step 3: truncation done — keeping {len(topk_docs)} doc(s) (top_k={topk})")

    if topk_docs:
        preview = ", ".join(
            f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})"
            for d in topk_docs[:3]
        )
        logger.debug(f"Step 3: top-3 preview: {preview}")

    return topk_docs


def node_rerank(state):
    """
    Rerank node: re-scores and truncates retrieved documents to improve answer relevance.
    """
    logger.info("--- node_rerank start ---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    doc_items = step_1_merge_docs(state)
    scored_docs = step_2_rerank_docs(state, doc_items)
    topk_docs = step_3_topk(scored_docs)

    logger.info(f"node_rerank done — {len(topk_docs)} doc(s) passed to answer node")

    add_done_task(
        state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    return {"reranked_docs": topk_docs}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> starting node_rerank test")
    print("=" * 50)

    mock_rrf_chunks = [
        {"entity": {"chunk_id": "local_1", "content": "RRF is a reciprocal rank fusion algorithm", "title": "Algorithm Introduction", "score": 0.9}},
        {"entity": {"chunk_id": "local_2", "content": "BGE is a powerful reranking model", "title": "Model Introduction", "score": 0.8}},
        {"entity": {"chunk_id": "local_3", "content": "Unrelated test document content", "title": "Test Document", "score": 0.1}},
    ]

    mock_web_docs = [
        {"title": "Rerank Explained", "url": "http://web.com/1", "snippet": "Reranking is the second-stage retrieval step commonly used in RAG systems"},
        {"title": "Unrelated Page", "url": "http://web.com/2", "snippet": "The weather is nice today, great for a walk"},
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "What are RRF and Rerank?",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }

    try:
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> test result summary:")
        print(f"total input docs: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"total output docs: {len(reranked)}")
        print("-" * 30)
        print("final ranking:")
        for i, doc in enumerate(reranked, 1):
            print(f"rank {i}: source={doc.get('source')}, score={doc.get('score'):.4f}, text={doc.get('text')[:20]}...")

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] rerank scoring is working")
        else:
            print("\n[FAIL] rerank scores are all zero or negative")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"unhandled exception during test: {e}")
