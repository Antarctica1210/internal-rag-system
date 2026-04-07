import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    Node: PDF convert to Markdown (node_pdf_to_md)
    Core task is to convert unstructured PDF data into structured Markdown data.
    Future implementations:
    1. Call MinerU (magic-pdf) tool.
    2. Convert PDF to Markdown format.
    3. Save the result to state["md_content"].
    """
    logger.info(f">>> [Stub] invoke node: {sys._getframe().f_code.co_name}")
    return state