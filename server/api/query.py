from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, List
from pydantic import BaseModel
import uuid
import time
import asyncio
import logging

from rag.memory_chain import get_conversational_rag_chain
from rag.streaming_chain import get_streaming_rag_chain, stream_rag_response
from db.mongo import save_chat, get_chat_history
from auth.dependencies import get_current_user_id
from storage.chat_storage import chat_storage

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Router
# -------------------------------------------------
router = APIRouter(prefix="/query", tags=["Query"])

# Request model
class QueryFilters(BaseModel):
    document_ids: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class SearchContext(BaseModel):
    """Web search context options."""
    enable_web_search: bool = False
    search_type: str = "hybrid"  # hybrid, web_only, perplexica
    max_web_results: int = 5
    # Perplexica options
    use_perplexica: bool = False
    perplexica_focus_mode: str = "webSearch"
    auto_web_threshold: float = 0.6

class QueryRequest(BaseModel):
    question: str
    filters: Optional[QueryFilters] = None
    conversation_id: Optional[str] = None
    search_context: Optional[SearchContext] = None


def _build_chat_history(previous_chats: List[Dict]) -> List[tuple[str, str]]:
    """Normalize mixed chat-history schemas into LangChain chat history tuples."""
    chat_history: List[tuple[str, str]] = []

    for chat in reversed(previous_chats):
        # Legacy schema: one record contains both question + answer
        if "question" in chat or "answer" in chat:
            question = chat.get("question")
            answer = chat.get("answer")
            if question:
                chat_history.append(("human", question))
            if answer:
                chat_history.append(("ai", answer))
            continue

        # New schema: one record per message
        message_type = chat.get("message_type")
        if message_type == "_metadata":
            continue

        content = chat.get("content")
        if not content:
            continue

        if message_type == "user":
            chat_history.append(("human", content))
        elif message_type == "assistant":
            chat_history.append(("ai", content))

    return chat_history

# -------------------------------------------------
# NORMAL QUERY (WITH MEMORY AND OPTIONAL WEB SEARCH)
# -------------------------------------------------
@router.post("/")
async def query_documents(
    request: QueryRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Stateless conversational RAG query with optional web search.
    Conversation history is passed explicitly.
    """
    question = request.question
    conversation_id = request.conversation_id
    search_context = request.search_context
    is_new_conversation = False
    start_time = time.time()
    web_sources = []

    # Create new conversation if not provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        is_new_conversation = True
        # Create conversation metadata
        chat_storage.create_conversation(user_id, question[:50])

    # Save user message
    chat_storage.save_message(
        user_id=user_id,
        conversation_id=conversation_id,
        message_type="user",
        content=question,
        conversation_title=question[:50] if is_new_conversation else None
    )

    # Check if web search is enabled
    web_context = ""
    perplexica_answer = ""
    
    # Perplexica search (if enabled)
    if search_context and search_context.use_perplexica:
        try:
            from services.perplexica_service import get_perplexica_service
            from config.perplexica import FocusMode
            
            service = get_perplexica_service()
            focus_mode = FocusMode(search_context.perplexica_focus_mode)
            
            result = await service.search(
                query=question,
                focus_mode=focus_mode,
                use_cache=True
            )
            
            if not result.error and result.sources:
                perplexica_answer = result.answer
                
                # Build web context from Perplexica
                web_parts = []
                for src in result.sources[:search_context.max_web_results]:
                    web_sources.append({
                        "source": "perplexica",
                        "title": src.title,
                        "url": src.url,
                        "snippet": src.snippet
                    })
                    web_parts.append(f"**{src.title}**\n{src.snippet}\nURL: {src.url}")
                
                web_context = "\n\n---\n\n".join(web_parts)
                logger.info(f"Added {len(result.sources)} Perplexica results")
                
        except Exception as e:
            logger.error(f"Perplexica search failed: {e}")
            # Fall back to MCP if available
    
    # MCP web search fallback (if Perplexica not used or failed)
    if search_context and search_context.enable_web_search and not web_sources:
        try:
            from mcp.mcp_client import get_mcp_client, format_web_results_for_context
            
            mcp_client = get_mcp_client()
            web_data = await mcp_client.search_and_extract(
                query=question,
                num_results=search_context.max_web_results
            )
            
            web_results = web_data.get("results", [])
            if web_results:
                web_context = await format_web_results_for_context(web_results)
                
                # Build web sources for response
                for result in web_results:
                    web_sources.append({
                        "source": "web",
                        "title": result.get("title"),
                        "url": result.get("url"),
                        "snippet": result.get("snippet", "")[:200]
                    })
                
                logger.info(f"Added {len(web_results)} web results to context")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            # Continue without web results

    # 1️⃣ Extract document filter IDs from request filters (if provided)
    filter_document_ids = None
    if request.filters and request.filters.document_ids:
        filter_document_ids = request.filters.document_ids
        logger.info(f"Document filter active: restricting to {filter_document_ids}")

    # 2️⃣ Build stateless chain with optional document filter
    chain = get_conversational_rag_chain(
        user_id,
        web_context=web_context,
        filter_document_ids=filter_document_ids
    )

    # 3️⃣ Fetch previous chat history from MongoDB
    previous_chats = get_chat_history(user_id, limit=6)

    # 3️⃣ Convert DB records → LangChain format
    chat_history = _build_chat_history(previous_chats)

    # 4️⃣ Call chain WITH chat_history
    try:
        response = chain({
            "question": question,
            "chat_history": chat_history
        })
    except Exception as e:
        logger.error(f"Chain error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save error message
        chat_storage.save_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_type="assistant",
            content=f"Error: {str(e)}",
            response_metadata={
                "status": "error",
                "error_message": str(e)
            }
        )
        raise HTTPException(status_code=500, detail=str(e))

    answer = response["answer"]
    response_time = int((time.time() - start_time) * 1000)

    # Build sources with full text (document sources)
    sources = []
    for doc in response.get("source_documents", []):
        sources.append({
            "source": "document",
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page"),
            "full_text": doc.page_content[:500] if doc.page_content else None,
            "relevance_score": doc.metadata.get("score", 0.85),
            "chunk_index": doc.metadata.get("chunk_index")
        })

    # Combine document and web sources
    all_sources = sources + web_sources

    # Save assistant message with sources
    chat_storage.save_message(
        user_id=user_id,
        conversation_id=conversation_id,
        message_type="assistant",
        content=answer,
        sources=all_sources,
        response_metadata={
            "model": "llama-3.3-70b-versatile",
            "response_time_ms": response_time,
            "status": "success",
            "source_count": len(all_sources),
            "doc_sources": len(sources),
            "web_sources": len(web_sources),
            "web_search_enabled": bool(search_context and search_context.enable_web_search)
        }
    )

    # 5️⃣ Persist this turn (for legacy compatibility)
    save_chat(
        user_id=user_id,
        question=question,
        answer=answer,
        sources=all_sources
    )

    return {
        "answer": answer,
        "sources": all_sources,
        "conversation_id": conversation_id,
        "is_new_conversation": is_new_conversation,
        "web_search_used": len(web_sources) > 0
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
# POST /query → normal RAG with memory and optional web search
# POST /query/stream → streaming response (Gemini)