import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Entry Point (node_entry)
    Why this name: As the Entry Point of the graph, it is responsible for receiving external input and determining the flow direction.
    Future implementations:
    1. Receive file path.
    2. Determine file type (PDF/MD).
    3. Set routing flags in state (is_pdf_read_enabled / is_md_read_enabled).
    """
    # mock entry point logic
    if path:= state.get("local_file_path"):
        if path.endswith(".pdf"):
            state["is_pdf_read_enabled"] = True
        elif path.endswith(".md"):
            state["is_md_read_enabled"] = True

    return state