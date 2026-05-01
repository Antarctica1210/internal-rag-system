import os
import sys
import time
import requests
import zipfile
import shutil
from pathlib import Path

from app.import_process.agent.state import ImportGraphState
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config
from app.core.logger import logger

MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key
MINERU_USE_LOCAL = mineru_config.use_local


# ---------------------------------------------------------------------------
# Step 1: Validate paths
# ---------------------------------------------------------------------------

def step_1_validate_paths(state: ImportGraphState):
    """
    Validate the PDF file path and output directory from workflow state.
    Creates the output directory if it does not exist.
    Returns: (pdf_path_obj: Path, output_dir_obj: Path)
    Raises: ValueError (missing params), FileNotFoundError (invalid file)
    """
    log_prefix = "[step_1_validate_paths]"
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()

    if not pdf_path:
        raise ValueError(f"{log_prefix} Missing workflow state param: pdf_path (current value: {repr(pdf_path)})")
    if not local_dir:
        raise ValueError(f"{log_prefix} Missing workflow state param: local_dir (current value: {repr(local_dir)})")

    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"{log_prefix} PDF file not found: {pdf_path_obj.absolute()}")
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"{log_prefix} Path is a directory, not a file: {pdf_path_obj.absolute()}")

    if not output_dir_obj.exists():
        logger.info(f"{log_prefix} Output directory does not exist, creating: {output_dir_obj.absolute()}")
        output_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, output_dir_obj


# ---------------------------------------------------------------------------
# Step 2 (Cloud): Upload to MinerU cloud and poll for result
# ---------------------------------------------------------------------------

def step_2_cloud_upload_and_poll(pdf_path_obj: Path) -> str:
    """
    Upload a PDF to the MinerU cloud API and poll until parsing completes.
    Returns: full_zip_url (str) — download URL for the result ZIP
    Raises: ValueError (missing config), RuntimeError (request/upload failure), TimeoutError
    """
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise ValueError("MinerU config missing: set MINERU_BASE_URL and MINERU_API_TOKEN in .env")
    logger.info(f"[cloud] Config validated, processing file: {pdf_path_obj.name}")

    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}",
    }

    # 1. Get signed upload URL and batch_id
    url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm",
    }
    logger.debug(f"[cloud] Requesting upload URL: {url_get_upload}, params: {req_data}")
    resp = requests.post(url=url_get_upload, headers=request_headers, json=req_data, timeout=90)

    if resp.status_code != 200:
        raise RuntimeError(f"[cloud] Failed to get upload URL, status: {resp.status_code}, body: {resp.text}")

    resp_data = resp.json()
    if resp_data["code"] != 0:
        raise RuntimeError(f"[cloud] API error getting upload URL: {resp_data}")

    signed_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]
    logger.info(f"[cloud] Upload URL received, batch_id: {batch_id}")

    # 2. Upload PDF binary to signed URL
    logger.info(f"[cloud] Reading PDF file: {pdf_path_obj.name}")
    with open(pdf_path_obj, "rb") as f:
        file_data = f.read()

    upload_session = requests.Session()
    upload_session.trust_env = False  # Disable proxy to avoid signed URL verification failure
    try:
        put_resp = upload_session.put(url=signed_url, data=file_data, timeout=60)
        if put_resp.status_code != 200:
            logger.warning(f"[cloud] First upload attempt failed (status: {put_resp.status_code}), retrying with explicit Content-Type")
            put_resp = upload_session.put(
                url=signed_url, data=file_data,
                headers={"Content-Type": "application/pdf"}, timeout=60
            )
            if put_resp.status_code != 200:
                raise RuntimeError(f"[cloud] Upload failed after retry, status: {put_resp.status_code}, body: {put_resp.text}")
        logger.info(f"[cloud] File uploaded successfully: {pdf_path_obj.name}")
    except Exception as e:
        raise RuntimeError(f"[cloud] Upload failed due to network error: {str(e)}")
    finally:
        upload_session.close()

    # 3. Poll batch status until done / failed / timeout
    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    start_time = time.time()
    timeout_seconds = 600
    poll_interval = 3
    logger.info(f"[cloud] Polling task status, batch_id: {batch_id}, timeout: {timeout_seconds}s")

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            raise TimeoutError(f"[cloud] Task timed out after {int(timeout_seconds)}s, batch_id: {batch_id}")

        try:
            poll_resp = requests.get(url=poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(f"[cloud] Poll request failed, retrying in {poll_interval}s: {str(e)}")
            time.sleep(poll_interval)
            continue

        if poll_resp.status_code != 200:
            if 500 <= poll_resp.status_code < 600:
                logger.warning(f"[cloud] Server busy (status: {poll_resp.status_code}), retrying in {poll_interval}s")
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[cloud] Poll request failed, status: {poll_resp.status_code}, body: {poll_resp.text}")

        poll_data = poll_resp.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"[cloud] API error during polling: {poll_data}")

        extract_results = poll_data["data"]["extract_result"]
        if not extract_results:
            logger.debug(f"[cloud] No result yet, elapsed: {int(elapsed_time)}s, waiting...")
            time.sleep(poll_interval)
            continue

        result_item = extract_results[0]
        state_status = result_item["state"]

        if state_status == "done":
            logger.info(f"[cloud] Task complete, elapsed: {int(elapsed_time)}s, batch_id: {batch_id}")
            full_zip_url = result_item.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError(f"[cloud] Task done but no ZIP URL returned, batch_id: {batch_id}")
            return full_zip_url
        elif state_status == "failed":
            err_msg = result_item.get("err_msg", "unknown error")
            raise RuntimeError(f"[cloud] Task failed, batch_id: {batch_id}, error: {err_msg}")
        else:
            logger.debug(f"[cloud] Processing (elapsed: {int(elapsed_time)}s), status: {state_status}")
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Step 2 (Local): Submit to local mineru-api and poll for result
# ---------------------------------------------------------------------------

def step_2_local_upload_and_poll(pdf_path_obj: Path, output_dir_obj: Path) -> Path:
    """
    Submit a PDF to the local self-hosted mineru-api, poll until complete,
    then save the result ZIP to disk.
    Returns: zip_save_path (Path) — path to the saved result ZIP file
    Raises: ValueError (missing config), RuntimeError (request failure), TimeoutError
    """
    if not MINERU_BASE_URL:
        raise ValueError("MinerU config missing: set MINERU_BASE_URL in .env (e.g. http://localhost:8000)")
    logger.info(f"[local] Submitting file to local API: {pdf_path_obj.name}")

    # 1. Submit file via async task endpoint
    tasks_url = f"{MINERU_BASE_URL}/tasks"
    with open(pdf_path_obj, "rb") as f:
        submit_resp = requests.post(
            url=tasks_url,
            files={"files": (pdf_path_obj.name, f, "application/pdf")},
            data={"return_md": "true"},
            lang_list=["en"],
            timeout=60,
        )

    if submit_resp.status_code != 200:
        raise RuntimeError(f"[local] Task submission failed, status: {submit_resp.status_code}, body: {submit_resp.text}")

    task_id = submit_resp.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"[local] No task_id in submission response: {submit_resp.json()}")
    logger.info(f"[local] Task submitted, task_id: {task_id}")

    # 2. Poll task status until done / failed / timeout
    status_url = f"{MINERU_BASE_URL}/tasks/{task_id}"
    result_url = f"{MINERU_BASE_URL}/tasks/{task_id}/result"
    start_time = time.time()
    timeout_seconds = 600
    poll_interval = 3
    logger.info(f"[local] Polling task status, task_id: {task_id}, timeout: {timeout_seconds}s")

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            raise TimeoutError(f"[local] Task timed out after {int(timeout_seconds)}s, task_id: {task_id}")

        try:
            poll_resp = requests.get(url=status_url, timeout=10)
        except Exception as e:
            logger.warning(f"[local] Poll request failed, retrying in {poll_interval}s: {str(e)}")
            time.sleep(poll_interval)
            continue

        if poll_resp.status_code != 200:
            logger.warning(f"[local] Poll returned status {poll_resp.status_code}, retrying in {poll_interval}s")
            time.sleep(poll_interval)
            continue

        status_data = poll_resp.json()
        task_status = status_data.get("status")

        if task_status == "done":
            logger.info(f"[local] Task complete, elapsed: {int(elapsed_time)}s, task_id: {task_id}")
            break
        elif task_status == "failed":
            err_msg = status_data.get("error", "unknown error")
            raise RuntimeError(f"[local] Task failed, task_id: {task_id}, error: {err_msg}")
        else:
            logger.debug(f"[local] Processing (elapsed: {int(elapsed_time)}s), status: {task_status}")
            time.sleep(poll_interval)

    # 3. Fetch result as ZIP and save to disk
    logger.info(f"[local] Fetching result ZIP for task_id: {task_id}")
    result_resp = requests.get(url=result_url, params={"response_format_zip": "true"}, timeout=120)
    if result_resp.status_code != 200:
        raise RuntimeError(f"[local] Failed to fetch result ZIP, status: {result_resp.status_code}, body: {result_resp.text}")

    zip_save_path = output_dir_obj / f"{pdf_path_obj.stem}_result.zip"
    with open(zip_save_path, "wb") as f:
        f.write(result_resp.content)
    logger.info(f"[local] Result ZIP saved: {zip_save_path}")
    return zip_save_path


# ---------------------------------------------------------------------------
# Step 3 (Cloud only): Download ZIP from URL
# ---------------------------------------------------------------------------

def step_3_download_zip(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> Path:
    """
    Download the result ZIP from a cloud URL and save to disk.
    Returns: zip_save_path (Path)
    """
    logger.info(f"[cloud] Downloading result ZIP from URL...")
    resp = requests.get(zip_url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"[cloud] ZIP download failed, status: {resp.status_code}")

    zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
    with open(zip_save_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"[cloud] ZIP saved: {zip_save_path}")
    return zip_save_path


# ---------------------------------------------------------------------------
# Step 4: Extract ZIP and locate the MD file
# ---------------------------------------------------------------------------

def step_4_extract_and_find_md(zip_save_path: Path, output_dir_obj: Path, pdf_stem: str) -> str:
    """
    Extract the result ZIP and locate the target MD file.
    Priority: same name as PDF > full.md > first .md found
    Returns: absolute path string to the final MD file
    Raises: FileNotFoundError (no .md file found)
    """
    extract_target_dir = output_dir_obj / pdf_stem
    if extract_target_dir.exists():
        try:
            shutil.rmtree(extract_target_dir)
            logger.info(f"[extract] Removed old extraction directory: {extract_target_dir}")
        except Exception as e:
            logger.warning(f"[extract] Failed to remove old directory, continuing anyway: {str(e)}")

    extract_target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
        zip_file_obj.extractall(extract_target_dir)
    logger.info(f"[extract] ZIP extracted to: {extract_target_dir}")

    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"[extract] No .md files found in extraction directory: {extract_target_dir}")
    logger.info(f"[extract] Found {len(md_file_list)} MD file(s), selecting by priority")

    # Priority 1: same stem as the PDF
    target_md_file = next((f for f in md_file_list if f.stem == pdf_stem), None)
    if target_md_file:
        logger.info(f"[extract] Priority 1 match (same name as PDF): {target_md_file.name}")

    # Priority 2: MinerU default output file
    if not target_md_file:
        target_md_file = next((f for f in md_file_list if f.name.lower() == "full.md"), None)
        if target_md_file:
            logger.info(f"[extract] Priority 2 match (full.md): {target_md_file.name}")

    # Priority 3: fallback to first found
    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.info(f"[extract] Priority 3 fallback (first file): {target_md_file.name}")

    # Rename to match PDF stem if needed
    if target_md_file.stem != pdf_stem:
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            target_md_file.rename(new_md_path)
            target_md_file = new_md_path
            logger.info(f"[extract] Renamed MD file to: {pdf_stem}.md")
        except OSError as e:
            logger.warning(f"[extract] Rename failed, using original name: {str(e)}")

    final_md_path = str(target_md_file.absolute())
    logger.info(f"[extract] Final MD file path: {final_md_path}")
    return final_md_path


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    Node: PDF to Markdown (node_pdf_to_md)
    Converts an unstructured PDF into structured Markdown via MinerU.
    Supports two modes controlled by MINERU_USE_LOCAL in .env:
      - false (default): MinerU cloud API (signed URL upload + batch polling)
      - true:            Local self-hosted mineru-api (POST /tasks + polling)
    """
    func_name = sys._getframe().f_code.co_name
    logger.debug(f"[{func_name}] Node started, state: {format_state(state)}")
    add_running_task(state["task_id"], func_name)

    try:
        # Step 1: validate paths
        pdf_path_obj, output_dir_obj = step_1_validate_paths(state)

        # Step 2 + 3: parse PDF and get ZIP path
        if MINERU_USE_LOCAL:
            logger.info(f"[{func_name}] Using local mineru-api: {MINERU_BASE_URL}")
            zip_save_path = step_2_local_upload_and_poll(pdf_path_obj, output_dir_obj)
        else:
            logger.info(f"[{func_name}] Using MinerU cloud API: {MINERU_BASE_URL}")
            zip_url = step_2_cloud_upload_and_poll(pdf_path_obj)
            zip_save_path = step_3_download_zip(zip_url, output_dir_obj, pdf_path_obj.stem)

        # Step 4: extract ZIP and locate MD file
        md_path = step_4_extract_and_find_md(zip_save_path, output_dir_obj, pdf_path_obj.stem)

        state["md_path"] = md_path
        logger.info(f"[{func_name}] MD file generated: {md_path}")

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
            logger.debug(f"[{func_name}] MD content loaded, length: {len(state['md_content'])} chars")
        except Exception as e:
            logger.error(f"[{func_name}] Failed to read MD file content: {str(e)}")

    except Exception as e:
        logger.error(f"[{func_name}] PDF to MD pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        add_done_task(state["task_id"], func_name)
        logger.debug(f"[{func_name}] Node finished, state: {format_state(state)}")

    return state


if __name__ == "__main__":
    logger.info("===== Starting node_pdf_to_md unit test =====")

    from app.import_process.agent.state import create_default_state
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"Project root: {PROJECT_ROOT}")

    test_pdf_path = os.path.join(PROJECT_ROOT, "test_doc", "The_IoT_and_AI_in_Agriculture_The_Time_Is_Now_A_Systematic_Review_of_Smart_Sensing_Technologies.pdf")
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)
    logger.info("===== node_pdf_to_md unit test complete =====")
