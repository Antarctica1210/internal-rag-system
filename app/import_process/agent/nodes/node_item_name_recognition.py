import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Item Name Recognition (node_item_name_recognition)
    Why this name: Recognize the core item/product name described in the document.
    Future implementations:
    1. Extract the first few paragraphs of the document.
    2. Call LLM to identify what this document is about (e.g., "Fluke 17B+ Multimeter").
    3. Store the result in state["item_name"] for subsequent data deduplication.
    """
    logger.info(f">>> [Stub] invoke node: {sys._getframe().f_code.co_name}")
    return state