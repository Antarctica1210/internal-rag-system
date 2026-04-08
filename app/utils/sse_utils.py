import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    READY = "ready"         # Connection established
    PROGRESS = "progress"   # Task node progress
    DELTA = "delta"         # LLM streaming output delta
    FINAL = "final"         # Final complete answer
    ERROR = "error"         # Error message
    CLOSE = "__close__"     # Close connection signal


# Global SSE session queue storage
# Key: session_id, Value: queue.Queue
_session_stream: Dict[str, queue.Queue] = {}

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """Get the queue for the specified session."""
    return _session_stream.get(session_id)

def create_sse_queue(session_id: str) -> "queue.Queue":
    """Create and register a new SSE queue for the specified session."""
    print(f"[SSE] Creating queue for session: {session_id}")
    q = queue.Queue()
    _session_stream[session_id] = q
    return q

def remove_sse_queue(session_id: str):
    """Remove the queue for the specified session."""
    print(f"[SSE] Removing queue for session: {session_id}")
    _session_stream.pop(session_id, None)

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """Pack data into SSE message format."""
    payload = json.dumps(data, ensure_ascii=False)
    # print(f"[SSE] Packing event: {event}, payload: {payload[:50]}...")
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    Push an event to the specified session by session_id.
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        # print(f"[SSE] Pushing to session {session_id}: {event}")
        stream_queue.put({"event": event, "data": data})
    else:
        print(f"[SSE] Warning: No queue found for session {session_id} when pushing {event}")

async def sse_generator(session_id: str, request: Request):
    """
    SSE generator for use with FastAPI's StreamingResponse.
    """
    print(f"[SSE] Generator started for session: {session_id}")
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # No matching queue found, exit immediately
        print(f"[SSE] Error: Queue not found for session {session_id}. Available sessions: {list(_session_stream.keys())}")
        return

    loop = asyncio.get_running_loop()
    try:
        # Send connection established signal
        print(f"[SSE] Sending ready signal for {session_id}")
        yield _sse_pack("ready", {})

        while True:
            # Exit as soon as the client disconnects
            if await request.is_disconnected():
                print(f"[SSE] Client disconnected: {session_id}")
                print("-----------------------Connection closed--------------------")
                break

            try:
                # Use run_in_executor to avoid blocking the async event loop
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                # print(f"[SSE] Queue empty for {session_id}, waiting...")
                continue

            event = msg.get("event")
            data = msg.get("data")

            # print(f"[SSE] Yielding event {event} for {session_id}")

            # Handle close signal
            if event == "__close__":
                print(f"[SSE] Closing signal received for {session_id}")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        print(f"[SSE] Client disconnected (Cancelled/Reset/Pipe): {session_id}")
        # Generator cancelled or peer disconnected: exit silently
        return
    except Exception as e:
        print(f"[SSE] Exception in generator for {session_id}: {e}")
    finally:
        print(f"[SSE] Generator finished for {session_id}")
        # Clean up resources
        remove_sse_queue(session_id)
