import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

_milvus_client = None


def get_milvus_client():
    """
    Get a singleton instance of MilvusClient.
    :return: MilvusClient instance if connection is successful, otherwise None
    """
    try:
        global _milvus_client

        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            if not milvus_uri:
                logger.error("Milvus client connection failed: MILVUS_URL environment variable is not configured")
                return None
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus client connected successfully")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus client connection error: {str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    Convert chunk_ids to INT64 type as required by the Milvus primary key schema.
    Filters out invalid IDs and separates convertible from non-convertible ones.
    :param ids: List of chunk_ids to convert
    :return: Tuple (ok_ids, bad_ids) — ok_ids are valid int64 IDs, bad_ids are unconvertible ones
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    Batch-fetch chunk data from Milvus by primary key (chunk_id).
    Used to hydrate chunks when only the chunk_id is known (no text content available).
    Prefers the get() method (direct primary key lookup, best performance);
    falls back to query() with a filter expression if get() fails.
    :param client: MilvusClient instance
    :param collection_name: Target collection name
    :param chunk_ids: List of chunk_ids to query
    :param output_fields: Fields to return; defaults to core chunk fields
    :param batch_size: Batch size to avoid oversized single queries; default 100
    :return: List[dict] of Milvus entity dicts; empty list on failure
    """
    if client is None:
        return []
    if not collection_name:
        return []
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"Some chunk_ids cannot be converted to INT64 and will be skipped: {bad_ids}")

    if not ok_ids:
        return []

    results = []
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # Preferred: direct primary key lookup via get() — best performance
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get() failed, falling back to query(): {str(e)}")

        # Fallback: filter-based query
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query() batch fetch by chunk_id failed: {str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    Build Milvus hybrid search request objects for dense and sparse vectors.
    :param dense_vector: Dense vector generated from the query text
    :param sparse_vector: Sparse vector generated from the query text
    :param dense_params: Dense vector search params; defaults to COSINE similarity
    :param sparse_params: Sparse vector search params; defaults to inner product (IP)
    :param expr: Filter expression for result narrowing
    :param limit: Number of results to return per vector search; default 5
    :return: List of ANN search requests [dense_req, sparse_req]
    """
    # COSINE matches the BGE-M3 dense index metric used at collection creation
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # IP matches the BGE-M3 sparse index metric used at collection creation
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    Execute a dense + sparse hybrid search on a Milvus collection.
    Uses WeightedRanker to fuse both vector results by configurable weights.
    :param client: MilvusClient instance
    :param collection_name: Target collection name
    :param reqs: Search request list — fixed order [dense_req, sparse_req]
    :param ranker_weights: Fusion weights for [dense, sparse]; default (0.5, 0.5)
    :param norm_score: Normalise scores to [0,1] before weighting to prevent scale mismatch
    :param limit: Final number of results to return; default 5
    :param output_fields: Fields to return; defaults to ["item_name"]
    :param search_params: Extra search params (e.g. ef, topk); default None
    :return: Hybrid search result list; None on failure
    """
    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        if output_fields is None:
            output_fields = ["item_name"]

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus hybrid search complete — collection [{collection_name}], {len(res[0])} result(s) returned")
        return res
    except Exception as e:
        logger.error(f"Milvus hybrid search failed — collection [{collection_name}]: {str(e)}", exc_info=True)
        return None
