from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict
from pydantic import BaseModel

from rag.memory_chain import get_conversational_rag_chain
from rag.streaming_chain import get_streaming_rag_chain, stream_rag_response
from db.mongo import save_chat, get_chat_history
from auth.dependencies import get_current_user_id

# -------------------------------------------------
# Router
# -------------------------------------------------
router = APIRouter(prefix="/query", tags=["Query"])

# Request model
class QueryRequest(BaseModel):
    question: str

# -------------------------------------------------
# NORMAL QUERY (WITH MEMORY)
# -------------------------------------------------
@router.post("/")
def query_documents(
    request: QueryRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Stateless conversational RAG query.
    Conversation history is passed explicitly.
    """
    question = request.question

    # 1️⃣ Build stateless chain (NO MEMORY INSIDE)
    chain = get_conversational_rag_chain(user_id)

    # 2️⃣ Fetch previous chat history from MongoDB
    previous_chats = get_chat_history(user_id, limit=6)

    # 3️⃣ Convert DB records → LangChain format
    chat_history = []
    for chat in reversed(previous_chats):
        chat_history.append(("human", chat["question"]))
        chat_history.append(("ai", chat["answer"]))

    # 4️⃣ Call chain WITH chat_history
    try:
        response = chain({
            "question": question,
            "chat_history": chat_history
        })
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

    # 5️⃣ Persist this turn
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
    request: QueryRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Streams answer token-by-token using Gemini.
    """
    return StreamingResponse(
        stream_rag_response(user_id, request.question),
        media_type="text/event-stream"
    )


# This file provides two endpoints:
# POST /query → normal RAG with memory
# POST /query/stream → streaming response (Gemini)