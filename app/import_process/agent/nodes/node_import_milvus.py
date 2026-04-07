import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Import to Milvus (node_import_milvus)
    Why this name: Write the processed vector data to the Milvus database.
    Future implementations:
    1. Connect to Milvus.
    2. Delete old data based on item_name (idempotency).
    3. Batch insert new vector data.
    """
    logger.info(f">>> [Stub] invoke node: {sys._getframe().f_code.co_name}")
    return state