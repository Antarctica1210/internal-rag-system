import os
import sys
from os.path import splitext

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    set_task_result,
    get_task_result,
    update_task_status,
    task_push_queue,
    clear_task,
)


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Entry Point (node_entry)
    Why this name: As the Entry Point of the graph, it is responsible for receiving external input and determining the flow direction.
    Future implementations:
    1. Receive file path.
    2. Determine file type (PDF/MD).
    3. Set routing flags in state (is_pdf_read_enabled / is_md_read_enabled).
    """
    # setup monitoring tools
    func_name = sys._getframe().f_code.co_name
    logger.debug(
        f"[{func_name}]node start，current workflow state: {format_state(state)}"
    )
    add_running_task(state["task_id"], func_name)

    # actual logic for the node
    document_path = state.get("local_file_path", "")
    if not document_path:
        logger.error(f"[{func_name}] Missing core parameter: local_file_path is empty")
        return state

    if document_path.endswith(".pdf"):
        logger.info(f"[{func_name}] Detected PDF file: {document_path}")
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = (
            document_path  # Store the PDF path in state for downstream nodes
        )
    elif document_path.endswith(".md"):
        logger.info(f"[{func_name}] Detected MD file: {document_path}")
        state["is_md_read_enabled"] = True
        state["md_path"] = (
            document_path  # Store the MD path in state for downstream nodes
        )
    else:
        logger.warning(f"[{func_name}] Unsupported file type for path: {document_path}")

    # extract file name for global identification (optional, can be used for logging or downstream processing)
    file_name = os.path.basename(document_path)
    state["file_title"] = splitext(file_name)[
        0
    ]  # Store the file title without extension in state
    logger.info(f"[{func_name}] Extracted file title: {state['file_title']}")

    # mark the node as done
    add_done_task(state["task_id"], func_name)

    return state
