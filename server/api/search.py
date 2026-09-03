"""
Search API for DocTalk

Provides hybrid search capabilities combining document search and web search.
"""

import logging
import time
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth.dependencies import get_current_user_id
from rag.retriever import get_retriever
from mcp.mcp_client import get_mcp_client
from mcp.mcp_server import check_mcp_health
from storage.search_cache import get_search_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


# Request/Response Models

class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"  # hybrid, docs_only, web_only
    top_k_docs: int = 5
    top_k_web: int = 5
    include_full_content: bool = True
    use_cache: bool = True


class HybridSearchResponse(BaseModel):
    query: str
    results: dict
    total_results: int
    processing_time_ms: int
    cached: bool = False


# Endpoints

@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(
    request: SearchRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Perform hybrid search across documents and web.
    
    Search types:
    - hybrid: Search both documents and web
    - docs_only: Search only uploaded documents
    - web_only: Search only the web
    """
    start_time = time.time()
    cache = get_search_cache()
    
    # Check cache first
    if request.use_cache:
        cached_result = cache.get(user_id, request.query, request.search_type)
        if cached_result:
            return HybridSearchResponse(
                query=request.query,
                results={
                    "documents": cached_result.get("results", {}).get("documents", []),
                    "web_results": cached_result.get("results", {}).get("web_results", []),
                    "merged_ranking": cached_result.get("results", {}).get("merged_ranking", [])
                },
                total_results=cached_result.get("total_results", 0),
                processing_time_ms=cached_result.get("processing_time_ms", 0),
                cached=True
            )
    
    doc_results = []
    web_results = []
    
    # Search documents
    if request.search_type in ["hybrid", "docs_only"]:
        try:
            retriever = get_retriever(user_id, k=request.top_k_docs)
            docs = retriever.get_relevant_documents(request.query)
            
            for i, doc in enumerate(docs):
                doc_results.append({
                    "source": "document",
                    "document_id": doc.metadata.get("file_id"),
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", 1),
                    "chunk_index": doc.metadata.get("chunk_index", i),
                    "text": doc.page_content[:500] if not request.include_full_content else doc.page_content,
                    "full_text": doc.page_content if request.include_full_content else None,
                    "relevance_score": doc.metadata.get("score", 0.8 - (i * 0.05))
                })
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            if request.search_type == "docs_only":
                raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")
    
    # Search web
    if request.search_type in ["hybrid", "web_only"]:
        try:
            mcp_client = get_mcp_client()
            
            if request.include_full_content:
                # Search and extract full content
                web_data = await mcp_client.search_and_extract(
                    query=request.query,
                    num_results=request.top_k_web
                )
            else:
                # Just search without extraction
                web_data = await mcp_client.web_search(
                    query=request.query,
                    num_results=request.top_k_web
                )
            
            for result in web_data.get("results", []):
                web_results.append({
                    "source": "web",
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "snippet": result.get("snippet"),
                    "full_content": result.get("full_content") if request.include_full_content else None,
                    "sections": result.get("sections", {}),
                    "relevance_score": 0.7 - (result.get("rank", 1) - 1) * 0.05
                })
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            if request.search_type == "web_only":
                raise HTTPException(status_code=500, detail=f"Web search failed: {str(e)}")
    
    # Merge and rank results
    merged_ranking = _merge_results(doc_results, web_results)
    
    processing_time = int((time.time() - start_time) * 1000)
    
    results = {
        "documents": doc_results,
        "web_results": web_results,
        "merged_ranking": merged_ranking
    }
    
    # Cache results
    if request.use_cache:
        cache.set(
            user_id=user_id,
            query=request.query,
            results=results,
            search_type=request.search_type,
            processing_time_ms=processing_time
        )
    
    return HybridSearchResponse(
        query=request.query,
        results=results,
        total_results=len(doc_results) + len(web_results),
        processing_time_ms=processing_time,
        cached=False
    )


def _merge_results(doc_results: List[dict], web_results: List[dict]) -> List[dict]:
    """
    Merge and rank document and web results.
    
    Uses interleaving with score-based ranking.
    """
    all_results = []
    
    # Add source type indicator and normalize scores
    for doc in doc_results:
        all_results.append({
            **doc,
            "combined_score": doc.get("relevance_score", 0.5) * 1.1  # Slight boost for docs
        })
    
    for web in web_results:
        all_results.append({
            **web,
            "combined_score": web.get("relevance_score", 0.5)
        })
    
    # Sort by combined score
    all_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    
    return all_results


@router.get("/cache")
async def get_cached_results(
    query: str = Query(None, description="Search text to filter cached queries"),
    limit: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user_id)
):
    """Get cached search results for the user."""
    cache = get_search_cache()
    
    if query:
        results = cache.search_cache(user_id, query, limit)
        return {"results": results, "total": len(results)}
    else:
        stats = cache.get_user_cache_stats(user_id)
        return stats


@router.delete("/cache/{query_hash}")
async def clear_cached_query(
    query_hash: str,
    user_id: str = Depends(get_current_user_id)
):
    """Clear a specific cached search result."""
    cache = get_search_cache()
    deleted = cache.delete(user_id, query_hash)
    
    if deleted:
        return {"message": "Cache entry deleted", "query_hash": query_hash}
    else:
        raise HTTPException(status_code=404, detail="Cache entry not found")


@router.delete("/cache")
async def clear_all_cache(
    confirm: bool = Query(False, description="Must be true to confirm deletion"),
    user_id: str = Depends(get_current_user_id)
):
    """Clear all cached search results for the user."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear all cache")
    
    cache = get_search_cache()
    deleted_count = cache.clear_user_cache(user_id)
    
    return {"message": f"Cleared {deleted_count} cached queries"}


@router.get("/mcp/health")
async def mcp_health():
    """Check MCP server health status."""
    health = await check_mcp_health()
    return health


@router.get("/mcp/tools")
async def get_mcp_tools():
    """Get list of available MCP tools."""
    from mcp.mcp_server import get_mcp_server
    server = get_mcp_server()
    return {"tools": server.get_tools()}
