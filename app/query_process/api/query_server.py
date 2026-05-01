import os
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT

from app.utils.task_utils import update_task_status, get_task_result, TASK_STATUS_PROCESSING
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator, push_to_session
from app.clients.mongo_history_utils import (
    get_recent_messages,
    clear_history
)
from app.query_process.agent.main_graph import get_query_app


# initialise FastAPI app
app = FastAPI(title="query service", description="Essay query service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    logger.info(f"Health check successful!")
    return {"status": "ok"}


@app.get("/chat.html")
async def chat():
    current_dir_parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chat_html_path = os.path.join(current_dir_parent_path, "page", "chat.html")
    if not chat_html_path.exists():
        logger.error(f"Page not exists: {chat_html_path}")
        raise HTTPException(status_code=404, detail=f"Page not exists: {chat_html_path}！")
    return FileResponse(chat_html_path)


class QueryRequest(BaseModel):
    query: str = Field(..., description="query content")
    session_id: str = Field(None, description="session ID")
    is_stream: bool = Field(False, description="whether to return results in a streaming fashion")


def run_query_graph(query: str, session_id: str, is_stream: bool):
    # 一会回调用 main_graph执行
    # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
    update_task_status(session_id, "processing", is_stream)

    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    try:
        query_app = get_query_app()
        query_app.invoke(state)
        # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
        # 修改 event = process
        update_task_status(session_id, "failed", is_stream)
        # 推送指定类型的事件
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.post("/query")  # 客户端 -》 问题 -》 graph开启了 -》 查到rag的结果 -》 返回即可！！
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)
        logger.info(f"[{session_id}] 已创建 SSE 消息队列")

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    logger.info(f"[{session_id}] 任务开始处理，查询内容：{user_query}")

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        logger.info(f"[{session_id}] 流式任务已提交至后台执行")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        logger.info(f"[{session_id}] 非流式查询处理完成")
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "done_list": []
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    logger.info(f"[{session_id}] 客户端已建立 SSE 流式连接")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count =  clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)