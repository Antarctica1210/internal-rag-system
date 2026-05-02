import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client, extract_response_text
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())

def step_3_extract_info(query, history) -> Dict:
    """
    Use an LLM to extract product name(s) (item_names) from the current query and conversation
    history. Returns an empty list when the product name cannot be determined. Also rewrites
    the question to be self-contained.
    :param query: current raw user query (e.g. "How much does this cost?")
    :param history: recent conversation history — list of dicts with role/text/_id fields
    :return: dict with two keys:
             {
                 "item_names": ["Product A", "Product B", ...],  # empty list if unknown
                 "rewritten_query": "Self-contained rewritten question"
             }
    """
    logger.info("Step 3: initialising LLM client...")
    client = get_llm_client(json_mode=True)

    # build history context text in "role: content" format
    history_text = ""
    for msg in history:
        history_text += f"{msg['role']}: {msg['text']}\n"
    logger.info(f"Step 3: history context ready (length: {len(history_text)})")

    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
    logger.info("Step 3: prompt loaded")

    messages = [
        SystemMessage(content="You are a professional customer service assistant skilled at understanding user intent and extracting key information."),
        HumanMessage(content=prompt)
    ]

    try:
        logger.info("Step 3: calling LLM...")
        response = client.invoke(messages)
        logger.info("Step 3: LLM response received")

        content = extract_response_text(response)
        # strip markdown code-block wrapper if present (e.g. ```json ... ```)
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)
        logger.info(f"Step 3: parsed LLM result: {result}")

        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query
        return result
    except Exception as e:
        logger.error(f"Step 3 LLM extraction failed: {e}")
        return {"item_names": [], "rewritten_query": query}


def step_4_vectorize_and_query(item_names) -> List[Dict]:
    """
    Embed each extracted item name (BGE-M3) and run a hybrid search against the Milvus
    collection to obtain similarity scores.
    :param item_names: list of product name strings extracted in step 3
    :return: list of dicts, one per product name:
             [
                 {
                     "extracted_name": "original extracted name",
                     "matches": [
                         {"item_name": "name stored in DB", "score": 0.98},
                         ...
                     ]
                 },
                 ...
             ]
    """
    logger.info(f"Step 4: starting vectorisation and query for items: {item_names}")
    results = []

    client = get_milvus_client()
    if not client:
        logger.error("Failed to connect to Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logging.error("No collection name found in env")
        return results

    # batch-generate BGE-M3 dense + sparse embeddings for all names at once
    logger.info("Step 4: generating embeddings...")
    embeddings = generate_embeddings(item_names)
    logger.info(f"Step 4: generated embeddings for {len(item_names)} item(s). Starting Milvus search...")

    for i in range(len(item_names)):
        try:
            logger.info(f"Step 4: processing item {i+1}/{len(item_names)}: {item_names[i]}")
            dense_vector = embeddings.get("dense")[i]
            sparse_vector = embeddings.get("sparse")[i]

            # build hybrid search request: [dense request, sparse request]
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=5
            )

            logger.info(f"Step 4: running hybrid search in collection '{collection_name}' for '{item_names[i]}'")
            # dense/sparse weight ratio 0.8/0.2; normalised scores (0-1)
            search_res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                limit=5,
                norm_score=True,
                output_fields=["item_name"]
            )
            logger.info(f"Step 4: '{item_names[i]}' search done. Found {len(search_res[0]) if search_res else 0} match(es).")

            matches = []
            if search_res and len(search_res) > 0:
                for hit in search_res[0]:
                    # hit format: {"id": ..., "distance": score, "entity": {"item_name": "..."}}
                    matches.append({
                        "item_name": hit.get("entity", {}).get("item_name"),
                        "score": hit.get("distance"),
                    })

            results.append({
                "extracted_name": item_names[i],
                "matches": matches
            })

        except Exception as e:
            logger.error(f"Step 4: error querying item name '{item_names[i]}': {e}")

    return results

def step_5_align_item_names(query_results) -> dict:
    """
    Align each extracted item name against Milvus search scores to produce confirmed and
    candidate name lists.
    Alignment rules (priority a > b > c > d):
        a  Exactly one match scores > 0.85  → confirm that name directly
        b  Multiple matches score > 0.85    → prefer the one equal to the extracted name;
                                              otherwise take the highest-scoring one
        c  No match scores > 0.85           → treat top-5 matches with score >= 0.6 as candidates
        d  No match scores >= 0.6           → return nothing (both lists empty)
    :param query_results: output of step 4
    :return: {
                 "confirmed_item_names": ["Name A", ...],  # deduplicated confirmed names
                 "options": ["Candidate A", ...]           # deduplicated candidate names
             }
    """
    confirmed_item_names: List[str] = []
    options: List[str] = []

    logger.info(f"Step 5: processing query results: {query_results}")

    for res in query_results:
        extracted_name = (res.get("extracted_name", "")).strip()
        matches = res.get("matches", []) or []
        if not matches:
            continue

        # sort descending by score so the highest-confidence match comes first
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) >= 0.6]

        # rule a: exactly one high-confidence match
        if len(high) == 1:
            confirmed_item_names.append(high[0].get("item_name"))
            continue

        # rule b: multiple high-confidence matches — prefer exact name match, else take top score
        if len(high) > 1:
            picked = None
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        break
            if not picked:
                picked = high[0]
            confirmed_item_names.append(picked.get("item_name"))
            continue

        # rule c: no high-confidence matches — take top-5 mid-confidence as candidates
        if len(mid) > 0:
            for m in mid[:5]:
                options.append(m.get("item_name"))

        # rule d: nothing qualifies — leave both lists unchanged

    return {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }

def step_6_check_confirmation(state, align_result, session_id, history, rewritten_query):
    """
    Check the alignment result from step 5 and update the graph state accordingly.
    Branch A — confirmed names found: update item_names and rewritten_query, clear any stale answer.
    Branch B — only candidates found: set a clarification question as the answer.
    Branch C — nothing found: set a "product not found" message as the answer.
    :param state: current graph state dict
    :param align_result: output of step 5
    :param session_id: session identifier
    :param history: recent conversation history
    :param rewritten_query: rewritten query from step 3
    :return: updated graph state dict
    """
    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    # branch A: high-confidence confirmed names
    if confirmed:
        # back-fill item_names on any history messages that lack them
        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))
        if ids_to_update:
            update_message_item_names(ids_to_update, confirmed)

        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state

    # branch B: only mid-confidence candidates — ask the user to clarify
    if options:
        options_str = ", ".join(options[:3])
        answer = f"Did you mean one of the following products: {options_str}? Please specify the exact model."
        state["answer"] = answer
        state["item_names"] = []
        return state

    # branch C: no matches at all
    state["answer"] = "Sorry, no matching product was found. Please provide an accurate model name so I can help you."
    state["item_names"] = []
    return state

def step_7_write_history(state, session_id, history, rewritten_query, message_id):
    """
    Persist the current turn to MongoDB.
    1. If the state contains an assistant answer (branches B/C), save it as an assistant message.
    2. Update the user's original message with the rewritten query and confirmed item names.
    :param state: graph state after step 6
    :param session_id: session identifier
    :param history: recent conversation history (reserved for future use)
    :param rewritten_query: rewritten query from step 3
    :param message_id: MongoDB document ID of the user message saved at the start of this node
    :return: state unchanged
    """
    if state.get("answer"):
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            rewritten_query="",
            item_names=state.get("item_names", [])
        )

    # update the user message saved at step 2 with the enriched rewritten_query and item_names
    save_chat_message(
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id
    )

    return state

def node_item_name_confirm(state):
    """
    Main node: product name confirmation pipeline.
    """
    logger.info(">>> node_item_name_confirm: start")

    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, "node_item_name_confirm", is_stream)

    # step 1: fetch recent history
    history = get_recent_messages(session_id, limit=10)
    logger.info(f"Node: fetched {len(history)} history message(s)")

    # step 2: save the user message now; step 7 will update it with enriched fields
    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Node: user message saved, ID: {message_id}")

    # step 3: extract item names and rewrite query via LLM
    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)
    state["rewritten_query"] = rewritten_query

    align_result = {}

    # steps 4 & 5: vectorise and align only when item names were extracted
    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: no item names extracted, skipping vector retrieval")

    # step 6: decide branch and update state
    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)

    # step 7: persist to history
    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)

    # carry history forward for downstream nodes (e.g. node_answer_output)
    final_state["history"] = history

    add_done_task(session_id, "node_item_name_confirm", is_stream)

    logger.info(f"Node: done. item_names={final_state.get('item_names')}")
    return final_state


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "How do I use the HAK 180 hot stamping machine?",
        "is_stream": False
    }

    print(">>> starting node_item_name_confirm test...")
    try:
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> test complete. final state:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        if result_state.get("item_names"):
            print(f"\n[PASS] confirmed item names: {result_state['item_names']}")
        else:
            print(f"\n[WARN] no item names confirmed (no vector match or LLM did not extract any)")

    except Exception as e:
        print(f"\n[FAIL] test error: {e}")
