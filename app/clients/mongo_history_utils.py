import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()


class HistoryMongoTool:
    """
    MongoDB chat history read/write utility (built on native PyMongo).
    Encapsulates connection setup, collection initialisation, and index creation.
    """
    def __init__(self):
        """
        Connect to MongoDB, obtain the database/collection, and create indexes.
        Raises on failure so callers are aware of connection problems.
        """
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            # compound index on (session_id asc, ts desc) for efficient per-session queries
            # create_index is idempotent — safe to call on every startup
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise


# module-level singleton; initialised eagerly so the first request is not slow
_history_mongo_tool = None
try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    # log a warning but do not crash — get_history_mongo_tool() will retry lazily
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")

def get_history_mongo_tool() -> HistoryMongoTool:
    """
    Return the singleton HistoryMongoTool instance, creating it lazily if needed.
    :return: shared HistoryMongoTool instance
    """
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool



def clear_history(session_id: str) -> int:
    """
    Delete all chat history for the given session.
    :param session_id: session identifier
    :return: number of documents deleted; 0 on failure
    """
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None
) -> str:
    """
    Insert or update a single chat message in MongoDB.
    When message_id is provided the existing document is updated; otherwise a new one is inserted.
    :param session_id: session identifier
    :param role: message role — "user" or "assistant"
    :param text: message content
    :param rewritten_query: rewritten query string used for retrieval (optional)
    :param item_names: associated product name list (optional)
    :param image_urls: associated image URL list (optional)
    :param message_id: document primary key — if given, update; otherwise insert
    :return: inserted ObjectId string on insert, or the passed message_id on update
    """
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "image_urls": image_urls,
        "ts": ts
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        # update existing document by primary key; $set leaves other fields untouched
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document}
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    Bulk-update the item_names field on the given message documents.
    :param ids: list of document primary key strings
    :param item_names: new product name list to set
    :return: number of documents modified; 0 on failure
    """
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": item_names}}
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch the most recent N messages for a session, ordered oldest-first.
    :param session_id: session identifier
    :param limit: maximum number of messages to return (default 10)
    :return: list of message dicts; empty list on failure
    """
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        return list(cursor)
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []


if __name__ == "__main__":
    sid = "000015_hybrid"
    save_chat_message(sid, "user", "Hello (Hybrid)")
    save_chat_message(sid, "assistant", "Hello! I am an assistant built on native Mongo + LangChain.")
    save_chat_message(sid, "user", "How do I replace the battery in this multimeter?", item_names=["Hybrid Multimeter"])

    print("--- query recent messages ---")
    messages = get_recent_messages(sid, limit=5)
    print(f"records found: {len(messages)}")
    for m in messages:
        print(f" {m}  ")
