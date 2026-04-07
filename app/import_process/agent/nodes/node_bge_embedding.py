import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    Node: BGE Embedding (node_bge_embedding)
    Why this name: Use the BGE-M3 model to convert text into embeddings.
    Future implementations:
    1. Load the BGE-M3 model.
    2. Perform dense (dense) and sparse (sparse) vectorization on the text of each Chunk.
    3. Prepare the data format for writing to Milvus.
    """
    logger.info(f">>> [Stub] invoke node: {sys._getframe().f_code.co_name}")
    return state