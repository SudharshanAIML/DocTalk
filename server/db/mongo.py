import os
import uuid
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv


# Load environment variables

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "docquery_ai")

if not MONGO_URI:
    raise ValueError("MONGODB_URI not set in environment variables")


# MongoDB Connection

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


# Collections

users_col = db["users"]
documents_col = db["documents"]
chunks_col = db["chunks"]
chat_history_col = db["chat_history"]


# Database & Index Initialization

def init_db():
    """
    Initialize collections and indexes.
    Safe to call multiple times.
    """
    users_col.create_index([("email", ASCENDING)], unique=True)

    documents_col.create_index([("user_id", ASCENDING)])
    documents_col.create_index([("file_id", ASCENDING)], unique=True)

    chunks_col.create_index([
        ("user_id", ASCENDING),
        ("faiss_index_id", ASCENDING)
    ])

    chat_history_col.create_index([
        ("user_id", ASCENDING),
        ("timestamp", DESCENDING)
    ])


# USER OPERATIONS

def create_user(email: str) -> dict:
    """
    Create a new user.
    """
    user = {
        "user_id": str(uuid.uuid4()),
        "email": email,
        "created_at": datetime.utcnow()
    }
    users_col.insert_one(user)
    return user


def get_user_by_email(email: str) -> dict | None:
    """
    Fetch user by email.
    """
    return users_col.find_one({"email": email}, {"_id": 0})


def get_user_by_id(user_id: str) -> dict | None:
    """
    Fetch user by user_id.
    """
    return users_col.find_one({"user_id": user_id}, {"_id": 0})



# DOCUMENT OPERATIONS

def insert_document(
    user_id: str,
    filename: str,
    file_type: str,
    num_pages: int
) -> dict:
    """
    Insert document metadata.
    """
    document = {
        "file_id": str(uuid.uuid4()),
        "user_id": user_id,
        "filename": filename,
        "file_type": file_type,
        "num_pages": num_pages,
        "uploaded_at": datetime.utcnow(),
        "status": "indexed"
    }
    documents_col.insert_one(document)
    return document


def get_user_documents(user_id: str) -> list:
    """
    Get all documents uploaded by a user.
    """
    return list(documents_col.find(
        {"user_id": user_id},
        {"_id": 0}
    ))


def get_document_by_id(user_id: str, file_id: str) -> dict | None:
    """
    Fetch a single document.
    """
    return documents_col.find_one(
        {"user_id": user_id, "file_id": file_id},
        {"_id": 0}
    )


def delete_document(user_id: str, file_id: str) -> None:
    """
    Delete document metadata.
    """
    documents_col.delete_one(
        {"user_id": user_id, "file_id": file_id}
    )



# CHUNK OPERATIONS (METADATA ONLY)

def insert_chunk(
    user_id: str,
    file_id: str,
    page_number: int,
    text_preview: str,
    faiss_index_id: int
) -> str:
    """
    Insert chunk metadata.
    """
    chunk_id = str(uuid.uuid4())
    chunk = {
        "chunk_id": chunk_id,
        "user_id": user_id,
        "file_id": file_id,
        "page_number": page_number,
        "text_preview": text_preview[:200],
        "faiss_index_id": faiss_index_id
    }
    chunks_col.insert_one(chunk)
    return chunk_id


def get_chunks_by_faiss_ids(user_id: str, faiss_ids: list[int]) -> list:
    """
    Retrieve chunk metadata by FAISS IDs.
    """
    return list(chunks_col.find(
        {
            "user_id": user_id,
            "faiss_index_id": {"$in": faiss_ids}
        },
        {"_id": 0}
    ))


def delete_chunks_by_file(user_id: str, file_id: str) -> None:
    """
    Delete all chunks for a document.
    """
    chunks_col.delete_many(
        {"user_id": user_id, "file_id": file_id}
    )



# CHAT HISTORY OPERATIONS

def save_chat(
    user_id: str,
    question: str,
    answer: str,
    sources: list
) -> None:
    """
    Save chat interaction.
    """
    chat = {
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.utcnow()
    }
    chat_history_col.insert_one(chat)


def get_chat_history(user_id: str, limit: int = 20) -> list:
    """
    Get recent chat history.
    """
    return list(chat_history_col.find(
        {"user_id": user_id},
        {"_id": 0}
    ).sort("timestamp", DESCENDING).limit(limit))


def delete_chat_history(user_id: str) -> None:
    """
    Delete user's chat history.
    """
    chat_history_col.delete_many({"user_id": user_id})



# DATABASE CLEANUP (OPTIONAL / ADMIN)

def delete_user_data(user_id: str) -> None:
    """
    Delete ALL data for a user.
    """
    documents_col.delete_many({"user_id": user_id})
    chunks_col.delete_many({"user_id": user_id})
    chat_history_col.delete_many({"user_id": user_id})
    users_col.delete_one({"user_id": user_id})



# DEBUG / HEALTH CHECK

def ping_db() -> bool:
    """
    Check database connection.
    """
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False
