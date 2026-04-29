from typing_extensions import TypedDict
from typing import List
import copy


class QueryGraphState(TypedDict):
    """
    Defines the data structure passed through the entire query pipeline.
    """

    session_id: str  # unique session identifier
    original_query: str  # raw user question

    # intermediate data from retrieval
    embedding_chunks: list  # chunks from standard vector search
    hyde_embedding_chunks: list  # chunks from HyDE vector search
    web_search_docs: list  # documents from web search

    # data from ranking stages
    rrf_chunks: list  # chunks after RRF fusion ranking
    reranked_docs: list  # final top-K documents after reranking

    # data for generation
    prompt: str  # assembled prompt
    answer: str  # final generated answer

    # auxiliary fields
    item_names: List[str]  # extracted product names
    rewritten_query: str  # rewritten user question
    history: list  # conversation history
    is_stream: bool  # whether to stream output


query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """
    Create the default query pipeline state with optional field overrides.
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    Deep-copy an existing state with optional field overrides; does not mutate the original.
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state
