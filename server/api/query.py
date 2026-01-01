from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict

from rag.memory_chain import get_conversational_rag_chain
from rag.streaming_chain import get_streaming_rag_chain
from db.mongo import save_chat

# -------------------------------------------------
# Router
# -------------------------------------------------
router = APIRouter(prefix="/query", tags=["Query"])

# -------------------------------------------------
# AUTH PLACEHOLDER (replace later with JWT)
# -------------------------------------------------
def get_current_user_id():
    return "test-user-id"

# -------------------------------------------------
# In-memory chain store (per user session)
# NOTE: Later you’ll replace this with Redis or DB
# -------------------------------------------------
user_memory_chains: Dict[str, object] = {}

# -------------------------------------------------
# NORMAL QUERY (WITH MEMORY)
# -------------------------------------------------
@router.post("/")
def query_documents(
    question: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Handles conversational RAG queries (non-streaming).
    """

    # Create chain once per user session
    if user_id not in user_memory_chains:
        user_memory_chains[user_id] = get_conversational_rag_chain(user_id)

    chain = user_memory_chains[user_id]

    try:
        response = chain({"question": question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    answer = response["answer"]

    sources = [
        {
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page")
        }
        for doc in response.get("source_documents", [])
    ]

    # Save chat to MongoDB
    save_chat(
        user_id=user_id,
        question=question,
        answer=answer,
        sources=sources
    )

    return {
        "answer": answer,
        "sources": sources
    }

# -------------------------------------------------
# STREAMING QUERY (NO MEMORY)
# -------------------------------------------------
@router.post("/stream")
def query_documents_stream(
    question: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Streams answer token-by-token using Gemini.
    """

    chain = get_streaming_rag_chain(user_id)

    def event_generator():
        try:
            # Streaming is handled internally by LangChain
            result = chain.run(question)
            yield result
        except Exception as e:
            yield f"\n[ERROR]: {str(e)}"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# This file provides two endpoints:
# POST /query → normal RAG with memory
# POST /query/stream → streaming response (Gemini)