import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Image Processing (node_md_img)
    Why this name: Process image resources (Image) in Markdown.
    Future implementations:
    1. Scan image links in Markdown.
    2. Upload images to MinIO object storage.
    3. (Optional) Call multimodal model to generate image descriptions.
    4. Replace image links in Markdown with MinIO URLs.
    """
    logger.info(f">>> [Stub] invoke node: {sys._getframe().f_code.co_name}")
    return state