import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import get_kb_import_workflow
from app.core.logger import logger


app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to the knowledge base (PDF/MD → parse → chunk → embed → Milvus)"
)

# Allow all origins for development; restrict to specific domains in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    """Serve the file import frontend page (import.html)."""
    html_abs_path = os.path.join(PROJECT_ROOT, "app", "import_process", "pages", "import.html")
    logger.info(f"Frontend page requested — path: {html_abs_path}")

    if not os.path.exists(html_abs_path):
        logger.error(f"Frontend page not found: {html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    return FileResponse(path=html_abs_path, media_type="text/html")


def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    Background task: execute the full LangGraph import pipeline for a single file.
    Task status transitions: pending → processing → completed / failed
    Each completed node is appended to the done_list so the frontend can poll progress.
    :param task_id: Unique task ID tied to this file's full pipeline run
    :param local_dir: Local directory for this task's intermediate files
    :param local_file_path: Absolute path of the uploaded file on disk
    """
    try:
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] LangGraph pipeline started — file: {local_file_path}")

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        kb_import_app = get_kb_import_workflow()
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                logger.info(f"[{task_id}] Node complete: {node_name}")
                add_done_task(task_id, node_name)

        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph pipeline finished successfully")

    except Exception as e:
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph pipeline failed: {str(e)}", exc_info=True)


@app.post(
    "/upload",
    summary="Upload files",
    description="Upload one or more files (PDF/MD) and trigger the knowledge base import pipeline for each."
)
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Receive uploaded files, save them locally, upload to MinIO, then kick off
    an independent LangGraph background task for each file.
    Files are stored under output/YYYYMMDD/<task_id>/ to avoid name collisions.
    :param background_tasks: FastAPI background task runner
    :param files: List of uploaded files (multipart/form-data)
    :return: JSON with task_ids for the caller to poll progress
    """
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    task_ids = []
    print(f"Received upload request — {len(files)} file(s), saving to: {date_based_root_dir}")
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] Processing upload — filename: {file.filename}, content-type: {file.content_type}")

        add_running_task(task_id, "upload_file")

        # Isolate each file under its own task directory to prevent name collisions
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)
        local_file_abs_path = os.path.join(task_local_dir, file.filename)

        with open(local_file_abs_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] File saved locally: {local_file_abs_path}")

        # Upload to MinIO for persistent storage; pipeline continues from local copy on failure
        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        try:
            minio_client = get_minio_client()
            if minio_client is None:
                raise HTTPException(
                    status_code=500,
                    detail="MinIO service connection failed, please check MinIO config"
                )
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type,
            )
            logger.info(f"[{task_id}] File uploaded to MinIO — bucket: {minio_bucket_name}, object: {minio_object_name}")
        except Exception as e:
            logger.warning(f"[{task_id}] MinIO upload failed — continuing with local file: {str(e)}", exc_info=True)

        add_done_task(task_id, "upload_file")

        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path)
        logger.info(f"[{task_id}] LangGraph background task queued")

    logger.info(f"Upload complete — {len(files)} file(s) queued, task IDs: {task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids,
    }


@app.get(
    "/status/{task_id}",
    summary="Query task status",
    description="Poll this endpoint with a task_id to get real-time processing progress."
)
async def get_task_progress(task_id: str):
    """
    Return the current status and node progress for a given task.
    All data is served from in-memory task state (task_utils.py) — no I/O overhead.
    :param task_id: Unique task ID returned by /upload
    :return: JSON with global status, completed nodes, and running nodes
    """
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }
    logger.info(
        f"[{task_id}] Status query — status: {task_status_info['status']}, done: {task_status_info['done_list']}"
    )
    return task_status_info


if __name__ == "__main__":
    logger.info("File Import Service starting...")
    # Use host="0.0.0.0" in production to accept external connections
    uvicorn.run(app=app, host="127.0.0.1", port=8000)
