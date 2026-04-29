import asyncio
import os
import json
import sys
from agents.mcp import MCPServerSse
from agents.mcp import MCPServerStreamableHttp
from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger


DASHSCOPE_BASE_URL_SSE = mcp_config.mcp_base_url
DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key


async def mcp_call(query):
    search_mcp = MCPServerSse(
        name="search_mcp",
        params={
            "url": DASHSCOPE_BASE_URL_SSE,
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300
        }
    )

    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5}
        )
        return result
    finally:
        await search_mcp.cleanup()

async def mcp_call_streamable(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300,
            "terminate_on_close": True,
        },
        max_retry_attempts=2,
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5},
        )
        return result
    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    logger.info("--- node_web_search_mcp start ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query", "")
    docs = []

    if not query:
        logger.warning("rewritten_query is empty, skipping web search")
    else:
        logger.info(f"running MCP web search for query: '{query}'")
        result = asyncio.run(mcp_call_streamable(query))
        if result:
            pages = json.loads(result.content[0].text).get("pages") or []
            logger.info(f"MCP returned {len(pages)} page(s)")
            # normalise into structured dicts for downstream reranking / citation
            for item in pages:
                snippet = (item.get("snippet") or "").strip()
                url = (item.get("url") or "").strip()
                title = (item.get("title") or "").strip()
                if not snippet:
                    continue
                docs.append({"title": title, "url": url, "snippet": snippet})
            logger.info(f"web search collected {len(docs)} valid doc(s)")
        else:
            logger.warning("MCP search returned no result")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("--- node_web_search_mcp end ---")
    if docs:
        return {"web_search_docs": docs}
    return {}