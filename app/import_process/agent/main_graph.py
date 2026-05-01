from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus

load_dotenv()

# ===================== 1. Initialise the graph =====================
workflow = StateGraph(ImportGraphState)

# ===================== 2. Register nodes =====================
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# ===================== 3. Set entry point =====================
workflow.add_edge(START, "node_entry")


# ===================== 4. Conditional routing after entry node =====================
def route_after_entry(state: ImportGraphState) -> str:
    """
    Route to the appropriate next node based on the input file type.
    PDF files go to node_pdf_to_md; MD files skip directly to node_md_img.
    """
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    elif state.get("is_md_read_enabled"):
        return "node_md_img"
    else:
        logger.error("No valid file type flag found in state after entry node — terminating workflow")
        return END


workflow.add_conditional_edges("node_entry", route_after_entry, {
    "node_pdf_to_md": "node_pdf_to_md",
    "node_md_img": "node_md_img",
    END: END
})

# ===================== 5. Set up edges =====================
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)


# ===================== 6. Compile the graph =====================
def get_kb_import_workflow():
    return workflow.compile()


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    import os

    logger.info("===== Knowledge base import end-to-end test started =====")

    test_pdf_name = os.path.join(
        "test_doc",
        "Sustainable_Development_of_AI_applications_in_Agriculture_A_Review.pdf"
    )
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)

    if not os.path.exists(test_pdf_path):
        logger.error(f"End-to-end test failed: test PDF not found at {test_pdf_path}")
        logger.info("Check the file path or place the test PDF in the project root test_doc directory")
    else:
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",
            "user_id": "test_user",
            "local_file_path": test_pdf_path,
            "local_dir": test_output_dir,
            "is_pdf_read_enabled": False,
            "is_md_read_enabled": False,
        })
        try:
            logger.info(f"Test task starting — PDF path: {test_pdf_path}")
            logger.info(f"Intermediate output directory: {test_output_dir}")
            logger.info("Executing full pipeline: entry → pdf2md → md_img → split → item_name → embedding → milvus")

            final_state = None
            kb_import_app = get_kb_import_workflow()
            for step in kb_import_app.stream(test_state, stream_mode="values"):
                current_node = list(step.keys())[-1] if step else "unknown"
                logger.info(f"Node complete: {current_node}")
                final_state = step

            if final_state:
                logger.info("-" * 80)
                logger.info("===== End-to-end test passed — result summary =====")
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_preview = final_state.get("md_content", "")[:150]
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False
                kg_id = final_state.get("kg_id", "not generated")

                logger.info(f"MD content preview (first 150 chars): {md_preview}...")
                logger.info(f"Total chunks generated: {chunk_count}")
                logger.info(f"All chunks vectorised: {'yes' if has_embedding else 'no'}")
                logger.info(f"All chunks saved to Milvus (with chunk_id): {'yes' if has_chunk_id else 'no'}")
                logger.info(f"Knowledge graph import ID: {kg_id}")
                logger.info(f"Final state keys: {list(final_state.keys())}")
                logger.info("-" * 80)
        except Exception as e:
            logger.error("===== End-to-end test failed =====", exc_info=True)
            logger.error(f"Error: {str(e)}")

    logger.info("===== Knowledge base import end-to-end test complete =====")
