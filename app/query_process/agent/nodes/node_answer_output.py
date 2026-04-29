import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "[[IMAGE_BLOCK]]"
MAX_CONTEXT_CHARS = 12000


def step_1_check_answer(state) -> bool:
    """
    Step 1: Check whether an answer already exists in state.
    - If yes: push it as a streaming delta (SSE) when needed, then return True.
    - If no:  return False.
    """
    answer = state.get("answer", None)
    is_stream = state.get("is_stream")
    if answer:
        if is_stream:
            logger.info("Step 1: pre-existing answer found — pushing as SSE delta")
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)
        return True
    else:
        return False


def step_2_construct_prompt(state: QueryGraphState) -> str:
    """
    Step 2: Build the answer prompt from reranked docs, conversation history,
    confirmed item names, and the (rewritten) user question.
    """
    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query", "")
    question = rewritten_query if rewritten_query else original_query
    history = state.get("history", [])
    item_names = state.get("item_names", [])
    reranked_docs = state.get("reranked_docs") or []

    # Build context string from reranked docs.
    # Each entry is formatted as a metadata header + body text, e.g.:
    #   "[1] [local] [chunk_id=123] [score=0.9500] [title=Manual]\n<content>"
    # Stop adding docs once MAX_CONTEXT_CHARS is reached to avoid token overflow.
    docs = []
    used = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        source = doc.get("source") or ""
        chunk_id = doc.get("chunk_id")
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()
        score = doc.get("score")

        meta_parts = [f"[{i}]"]
        if source:
            meta_parts.append(f"[{source}]")
        if chunk_id:
            meta_parts.append(f"[chunk_id={chunk_id}]")
        if url:
            meta_parts.append(f"[url={url}]")
        if score is not None:
            meta_parts.append(f"[score={float(score):.4f}]")
        if title:
            meta_parts.append(f"[title={title}]")
            doc = " ".join(meta_parts) + "\n" + text
        if used + len(doc) > MAX_CONTEXT_CHARS:
            break
        docs.append(doc)
        used += len(doc) + 2  # +2 for the "\n\n" separator
    context_str = "\n\n".join(docs) if docs else "No reference content available"

    # Build history string.
    # `used` carries over from the context block — if docs already consumed most of the
    # budget, history may be truncated or omitted entirely.
    history_str = ""
    if history:
        for msg in history:
            # MongoDB stores messages as {"role": "user"/"assistant", "text": "..."}
            role = msg.get("role")
            text = msg.get("text")
            if role == "user" and text:
                history_str += f"User: {text}\n"
            elif role == "assistant" and text:
                history_str += f"Assistant: {text}\n"

            used += len(history_str) + 2
            if used > MAX_CONTEXT_CHARS:
                break
    else:
        history_str = "No conversation history"

    item_names_str = ", ".join(item_names) if item_names else "No specific product"

    prompt = load_prompt(
        "answer_out",
        context=context_str,
        history=history_str,
        item_names=item_names_str,
        question=question,
    )

    logger.info(f"prompt constructed: {prompt}")
    return prompt


def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
    """
    Step 3: Call the LLM to generate an answer; supports both streaming and blocking modes.
    """
    logger.info("Step 3: generating answer (LLM)")
    logger.debug(f"final prompt: {prompt}")

    llm = get_llm_client()
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")

    if is_stream:
        logger.info(f"mode: streaming, session={session_id}")
        final_text = ""
        try:
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    final_text += delta
                push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})

            logger.info(f"streaming complete, total length: {len(final_text)}")

        except Exception as e:
            logger.error(f"streaming generation error: {e}", exc_info=True)
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

        state["answer"] = final_text
    else:
        logger.info(f"mode: blocking, session={session_id}")
        try:
            response = llm.invoke(prompt)
            content = response.content
            state["answer"] = content
            set_task_result(session_id, "answer", content)
            logger.info(f"answer generated, length: {len(content)}")
        except Exception as e:
            logger.error(f"answer generation error: {e}", exc_info=True)
            state["answer"] = "Sorry, an error occurred while generating the answer."

    return state


def _extract_images_from_docs(docs):
    """
    Extract unique image URLs from a list of document dicts.

    Strategy 1: check the doc's "url" field — used for web search results.
                 Keep only URLs with a recognised image extension.
    Strategy 2: scan the doc's "text" field for Markdown image syntax
                 ![alt](url) — used for local KB chunks (Markdown source).

    Regex: r'!\\[.*?\\]\\((.*?)\\)'
      - .*?   non-greedy alt text inside []
      - (.*?) capture group — the URL inside ()
      findall returns only the captured group, i.e. the URL string.

    :param docs: list of document dicts
    :return: deduplicated list of image URL strings
    """
    images = []
    seen = set()
    if not docs:
        return []

    md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    logger.info(f"extracting images from {len(docs)} doc(s)")

    for i, doc in enumerate(docs):
        # strategy 1: direct URL field (web search results)
        url = (doc.get("url") or "").strip()
        if url and url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
            if url not in seen:
                logger.debug(f"doc[{i}] image URL from field: {url}")
                seen.add(url)
                images.append(url)

        # strategy 2: Markdown image syntax in text (local KB chunks)
        text = (doc.get("text") or "").strip()
        if text:
            for img_url in md_img_pattern.findall(text):
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    logger.debug(f"doc[{i}] image URL from markdown: {img_url}")
                    seen.add(img_url)
                    images.append(img_url)

    logger.info(f"image extraction done — {len(images)} unique image(s): {images}")
    return images


def step_4_write_history(state: QueryGraphState, image_urls=None) -> QueryGraphState:
    """
    Step 4: Persist the assistant answer for this turn to MongoDB history.
    Failures are caught and logged without interrupting the main pipeline.
    """
    session_id = state.get("session_id", "default")
    answer = (state.get("answer") or "").strip()
    item_names = state.get("item_names") or []

    try:
        if answer:
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query="",
                item_names=item_names,
                image_urls=image_urls,
                message_id=None,
            )
    except Exception as e:
        logger.error(f"failed to write answer to MongoDB history: {e}")

    return state


def node_answer_output(state):
    """
    Answer output node.
    1. If an answer already exists in state (e.g. clarification / rejection from item
       confirm node), push it directly and skip LLM generation.
    2. Otherwise build the prompt from reranked docs, history, and item names,
       then call the LLM (streaming or blocking).
    3. Extract image URLs from the reranked docs.
    4. Persist the assistant answer to MongoDB history.
    5. Push a FINAL SSE event so the frontend can render images and close the stream.
    """
    logger.info("--- node_answer_output start ---")
    add_running_task(
        state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # step 1: use pre-existing answer if available
    answer_exists = step_1_check_answer(state)

    if not answer_exists:
        # step 2: build prompt
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt

        # step 3: generate answer via LLM
        step_3_generate_response(state, prompt)

    # extract images for history and frontend display
    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])

    # step 4: persist answer to MongoDB
    if state.get("answer"):
        logger.info("writing answer to MongoDB history")
        step_4_write_history(state, image_urls=image_urls)

    add_done_task(
        state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # step 5: send FINAL event to trigger frontend image rendering and stream close
    logger.info(f"pushing FINAL event with {len(image_urls)} image(s)")
    if state.get("is_stream"):
        push_to_session(
            state['session_id'],
            SSEEvent.FINAL,
            {
                "answer": state["answer"],
                "status": "completed",
                "image_urls": image_urls,
            },
        )

    logger.info("--- node_answer_output end ---")
    return state
