import copy
from typing import TypedDict

from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    State of the import graph, used to track the progress and status of the import process.
    """
    task_id: str

    # process identification
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # chunk processing
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # file path
    local_dir: str
    local_file_path: str
    file_title: str
    pdf_path: str
    md_path: str
    split_path: str
    embeddings_path: str

    # file content
    md_content: str
    chunks: list
    item_name: str

    # database content
    embeddings_content: list

# initialise a default state for the import graph
graph_default_state = ImportGraphState(
    task_id="",
    is_md_read_enabled=False,
    is_pdf_read_enabled=False,
    is_normal_split_enabled=True,
    is_silicon_flow_api_enabled=True,
    is_advanced_split_enabled=False,
    is_vllm_enabled=False,
    local_dir="",
    local_file_path="",
    file_title="",
    pdf_path="",
    md_path="",
    split_path="",
    embeddings_path="",
    md_content="",
    chunks=[],
    item_name="",
    embeddings_content=[]
)

def create_default_state(**overrides) -> ImportGraphState:
    """Create a default state for the import graph, with optional overrides for specific fields.
    This function creates a new state based on the default state, and allows overriding specific fields
    """
    state = copy.deepcopy(graph_default_state)

    state.update(overrides)
    return state

def get_default_state() -> ImportGraphState:
    """Get a copy of the default state for the import graph. This ensures that each import process starts with a clean state."""
    return copy.deepcopy(graph_default_state)