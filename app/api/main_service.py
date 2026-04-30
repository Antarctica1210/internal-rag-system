import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger

from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
    get_task_result,
    TASK_STATUS_PROCESSING,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import get_kb_import_workflow

from app.query_process.agent.state import create_query_default_state
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator, push_to_session
from app.clients.mongo_history_utils import get_recent_messages, clear_history
from app.query_process.agent.main_graph import get_query_app


app = FastAPI(
    title="Internal RAG Service",
    description="Combined import and query service — upload documents and query the knowledge base.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── Import pipeline ──────────────────────────────────────────────────────────

@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    """Serve the file import UI."""
    html_path = str(PROJECT_ROOT / "app" / "import_process" / "pages" / "import.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="import.html not found")
    return FileResponse(path=html_path, media_type="text/html")


def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """Background task: run the full LangGraph import pipeline for one file."""
    try:
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] import pipeline started — file: {local_file_path}")

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        kb_import_app = get_kb_import_workflow()
        for event in kb_import_app.stream(init_state):
            for node_name, _ in event.items():
                logger.info(f"[{task_id}] node complete: {node_name}")
                add_done_task(task_id, node_name)

        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] import pipeline finished successfully")
    except Exception as e:
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] import pipeline failed: {e}", exc_info=True)


@app.post("/upload", summary="Upload files for import")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Receive PDF/MD files, save locally, upload to MinIO, then queue a
    LangGraph background task per file. Poll /status/{task_id} for progress.
    """
    date_dir = str(PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d"))
    task_ids = []
    logger.info(f"upload request — {len(files)} file(s)")

    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] uploading — filename: {file.filename}")

        add_running_task(task_id, "upload_file")

        task_local_dir = os.path.join(date_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)
        local_file_abs_path = os.path.join(task_local_dir, file.filename)

        with open(local_file_abs_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"[{task_id}] file saved: {local_file_abs_path}")

        minio_object_name = (
            f"{os.getenv('MINIO_PDF_DIR', 'pdf_files')}"
            f"/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        )
        try:
            minio_client = get_minio_client()
            if minio_client is None:
                raise HTTPException(status_code=500, detail="MinIO connection failed")
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type,
            )
            logger.info(f"[{task_id}] uploaded to MinIO: {minio_bucket_name}/{minio_object_name}")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"[{task_id}] MinIO upload failed, continuing with local file: {e}")

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path)
        logger.info(f"[{task_id}] background task queued")

    return {
        "code": 200,
        "message": f"{len(files)} file(s) queued for import",
        "task_ids": task_ids,
    }


@app.get("/status/{task_id}", summary="Poll import task progress")
async def get_task_progress(task_id: str):
    """Return the current status and node progress for a given import task."""
    info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }
    logger.info(f"[{task_id}] status: {info['status']}, done: {info['done_list']}")
    return info


# ─── Query pipeline ───────────────────────────────────────────────────────────

@app.get("/chat.html")
async def chat():
    """Serve the chat UI."""
    chat_html_path = str(PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html")
    if not os.path.exists(chat_html_path):
        raise HTTPException(status_code=404, detail=f"Page not found: {chat_html_path}")
    return FileResponse(chat_html_path)


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    session_id: str = Field(None, description="Session ID (auto-generated if omitted)")
    is_stream: bool = Field(False, description="Enable SSE streaming")


def run_query_graph(query: str, session_id: str, is_stream: bool):
    """Background task: run the full LangGraph query pipeline."""
    update_task_status(session_id, "processing", is_stream)
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream,
    )
    try:
        query_app = get_query_app()
        query_app.invoke(state)
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(f"[{session_id}] query pipeline error: {e}")
        update_task_status(session_id, "failed", is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Submit a question to the RAG pipeline.
    - Blocking  (is_stream=false): waits and returns the full answer.
    - Streaming (is_stream=true):  returns immediately; connect /stream/{session_id} for SSE.
    """
    user_query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)
        logger.info(f"[{session_id}] SSE queue created")

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    logger.info(f"[{session_id}] query received: {user_query}")

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        logger.info(f"[{session_id}] streaming task submitted to background")
        return {
            "message": "Processing — connect to /stream/{session_id} for results.",
            "session_id": session_id,
        }
    else:
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        logger.info(f"[{session_id}] blocking query completed")
        return {
            "message": "completed",
            "session_id": session_id,
            "answer": answer,
            "done_list": [],
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE endpoint — streams DELTA and FINAL events for a query session."""
    logger.info(f"[{session_id}] SSE connection established")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """Return the most recent `limit` messages for a session."""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = [
            {
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear all history for a session."""
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    logger.info("Combined RAG service starting on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
