from langgraph.graph import StateGraph, END

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import (
    node_search_embedding_hyde,
)
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

builder = StateGraph(QueryGraphState)

# register nodes
builder.add_node("node_item_name_confirm", node_item_name_confirm)
builder.add_node("node_search_embedding", node_search_embedding)
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)
builder.add_node("node_rerank", node_rerank)
builder.add_node("node_answer_output", node_answer_output)

# entry point
builder.set_entry_point("node_item_name_confirm")


# conditional routing
def route_after_item_confirm(state: QueryGraphState):
    if state.get("answer"):
        # node_item_name_confirm sets `answer` directly in two cases:
        # 1. Ambiguous product — multiple candidates found; the node returns a clarification
        #    question asking the user to pick one (e.g. "Did you mean Model A or Model B?").
        # 2. Product not found — no match in the DB or confidence < 0.6; the node returns a
        #    rejection message. In both cases skip retrieval and go straight to output.
        return "node_answer_output"
    # fan-out to all three parallel search nodes
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
    },
)

# all three search nodes converge into RRF
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")

# normal pipeline
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)


def get_query_app():
    return builder.compile()


if __name__ == "__main__":
    import json

    from app.query_process.agent.main_graph import get_query_app
    from app.query_process.agent.state import create_query_default_state
    from app.core.logger import logger

    logger.info("===== test start =====")

    initial_state = create_query_default_state(
        session_id="test_001", original_query="How is the Huawei P60?"
    )
    final_state = None

    # stream output: only the final state dict per node, no metadata or execution logs
    query_app = get_query_app()
    for event in query_app.stream(initial_state):
        for key, value in event.items():
            logger.info(f"node: {key}")
            final_state = value

    logger.info(f"final state: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

    logger.info("graph structure:")
    # uv add grandalf
    query_app.get_graph().print_ascii()

    logger.info("===== test end =====")
