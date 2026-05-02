import os
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import update_task_status, get_task_result, TASK_STATUS_PROCESSING
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator, push_to_session
from app.clients.mongo_history_utils import (
    get_recent_messages,
    clear_history
)
from app.query_process.agent.main_graph import get_query_app


# initialise FastAPI app
app = FastAPI(title="query service", description="Essay query service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    logger.info(f"Health check successful!")
    return {"status": "ok"}


@app.get("/chat.html")
async def chat():
    current_dir_parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chat_html_path = os.path.join(current_dir_parent_path, "page", "chat.html")
    if not os.path.exists(chat_html_path):
        logger.error(f"Page not found: {chat_html_path}")
        raise HTTPException(status_code=404, detail=f"Page not found: {chat_html_path}")
    return FileResponse(chat_html_path)


class QueryRequest(BaseModel):
    query: str = Field(..., description="query content")
    session_id: str = Field(None, description="session ID")
    is_stream: bool = Field(False, description="whether to return results in a streaming fashion")


def run_query_graph(query: str, session_id: str, is_stream: bool):
    update_task_status(session_id, "processing", is_stream)

    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    try:
        query_app = get_query_app()
        query_app.invoke(state)
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(f"[{session_id}] query pipeline error: {str(e)}")
        update_task_status(session_id, "failed", is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    1. Parse request parameters.
    2. Update task status to processing.
    3. Run the query graph (background for streaming, blocking otherwise).
    4. Return the result or a processing acknowledgement.
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)
        logger.info(f"[{session_id}] SSE queue created")

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    logger.info(f"[{session_id}] task started, query: {user_query}")

    if is_stream:
        background_tasks.add_task(run_query_graph, user_query, session_id, is_stream)
        logger.info(f"[{session_id}] streaming task submitted to background")
        return {
            "message": "Processing, connect to /stream/{session_id} for results.",
            "session_id": session_id
        }
    else:
        run_query_graph(user_query, session_id, is_stream)
        answer = get_task_result(session_id, "answer", "")
        logger.info(f"[{session_id}] blocking query completed")
        return {
            "message": "completed",
            "session_id": session_id,
            "answer": answer,
            "done_list": []
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    logger.info(f"[{session_id}] SSE stream connection established")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """Return the most recent `limit` messages for a session."""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count =  clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)