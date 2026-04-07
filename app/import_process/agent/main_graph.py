from dotenv import load_dotenv
# Load LangGraph dependencies:StateGraph, START/END constants
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# 导入自定义状态类：统一管理工作流全程的所有数据（各节点共享/修改）
from app.import_process.agent.state import ImportGraphState, create_default_state
# 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库

load_dotenv()

# ===================== 1. Initialize the Graph =====================
workflow = StateGraph(ImportGraphState)

# ===================== 2. register nodes =====================
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# ===================== 3. set entry point =====================
workflow.add_edge(START, "node_entry")

# ===================== 4. define conditional routing function (branching logic after entry node) =====
def route_after_entry(state: ImportGraphState) -> str:
    """
    Routing function after entry node: determine the next node based on file type.
    If it's a PDF file, go to node_pdf_to_md; if it's an MD file, skip directly to node_md_img.
    """
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    elif state.get("is_md_read_enabled"):
        return "node_md_img"
    else:
        logger.error("No valid file type flag found in state after entry node.")
        return END  # No valid route, end the workflow
    
# set conditional edge from entry node to the next nodes
workflow.add_conditional_edges("node_entry", route_after_entry, {
    "node_pdf_to_md": "node_pdf_to_md",
    "node_md_img": "node_md_img",
    END: END
})

# ===================== 5. setup edges =====================
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# ===================== 6. compile the graph =====================
def get_kb_import_workflow():
    return workflow.compile()