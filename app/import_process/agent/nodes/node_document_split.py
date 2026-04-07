import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Document Splitting (node_document_split)
    Why this name: Split long documents into smaller Chunks (slices) for easier retrieval.
    Future implementations:
    1. Recursively split based on Markdown heading levels.
    2. Perform secondary splitting on overly long paragraphs.
    3. Generate a list of Chunks containing Metadata (heading paths).
    """
    logger.info(f">>> [Stub] invoke node: {sys._getframe().f_code.co_name}")
    return state