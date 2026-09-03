"""
Deep Search API Endpoints.

Provides endpoints for:
- POST /search/deep - Main deep search endpoint
- POST /search/deep/reasoning - Get detailed reasoning steps
- GET /search/relationships - Get document/concept relationships
- POST /search/cross-document - Cross-document analysis
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import time

from auth.dependencies import get_current_user_id
from db.neo4j_client import ping_neo4j
from rag.deep_search import (
    DeepSearchEngine, DeepSearchConfig, SearchDepth,
    get_deep_search_engine, deep_search
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Deep Search"])


# ==================== Request/Response Models ====================

class DeepSearchRequest(BaseModel):
    """Request body for deep search."""
    query: str = Field(..., description="Search query", min_length=1)
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    depth: str = Field("moderate", description="Search depth: simple, moderate, deep")
    include_reasoning: bool = Field(True, description="Include reasoning chain in response")
    cross_document: bool = Field(True, description="Enable cross-document analysis")
    context_limit: int = Field(20000, description="Max context tokens", ge=1000, le=100000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do the principles in chapter 2 apply to the case study in chapter 5?",
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "depth": "moderate",
                "include_reasoning": True,
                "cross_document": True,
                "context_limit": 20000
            }
        }


class ReasoningStep(BaseModel):
    """A step in the reasoning chain."""
    step: int
    question: str
    answer: str
    sources: List[dict] = []
    confidence: float = 0.0


class DocumentRelationship(BaseModel):
    """A relationship between documents or concepts."""
    source: str
    target: str
    type: str
    strength: float
    evidence: List[str] = []


class Source(BaseModel):
    """A source document/chunk."""
    source: str = "document"
    filename: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    relevance_score: float = 0.0
    full_text: Optional[str] = None
    matched_concepts: List[str] = []


class DeepSearchResponse(BaseModel):
    """Response from deep search."""
    answer: str
    summary: str
    reasoning_chain: Optional[List[ReasoningStep]] = None
    document_relationships: List[DocumentRelationship] = []
    confidence: float
    completeness: float
    processing_time_ms: int
    sources: List[Source] = []
    follow_up_suggestions: List[str] = []
    related_concepts: List[str] = []
    query_analysis: dict = {}


class ReasoningRequest(BaseModel):
    """Request for detailed reasoning."""
    query: str = Field(..., min_length=1)
    depth: str = Field("moderate")


class ReasoningResponse(BaseModel):
    """Response with detailed reasoning."""
    query: str
    steps: List[ReasoningStep]
    final_answer: str
    total_confidence: float
    processing_time_ms: int


class RelationshipQuery(BaseModel):
    """Query for relationships."""
    concept: Optional[str] = None
    document_id: Optional[str] = None


class CrossDocumentRequest(BaseModel):
    """Request for cross-document search."""
    query: str = Field(..., min_length=1)
    document_ids: Optional[List[str]] = None


class CrossDocumentResponse(BaseModel):
    """Response from cross-document search."""
    query_concepts: List[str]
    related_documents: List[str]
    document_similarities: List[dict]
    concept_relationships: List[dict]


# ==================== Endpoints ====================

@router.post("/deep", response_model=DeepSearchResponse)
async def perform_deep_search(
    request: DeepSearchRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Perform a deep search with multi-hop reasoning.
    
    This endpoint provides:
    - Query understanding and complexity analysis
    - Multi-layer retrieval (vector + semantic)
    - Chain-of-thought reasoning for complex queries
    - Cross-document relationship detection
    - Intelligent result ranking and synthesis
    
    **Depth Levels:**
    - `simple`: Fast search, minimal reasoning (~1s)
    - `moderate`: Balanced search with reasoning (~3-5s)
    - `deep`: Thorough analysis with full reasoning (~5-15s)
    """
    start_time = time.time()
    
    try:
        # Validate depth
        try:
            depth = SearchDepth(request.depth)
        except ValueError:
            depth = SearchDepth.MODERATE
        
        # Build config
        config = DeepSearchConfig(
            depth=depth,
            max_context_tokens=request.context_limit,
            include_reasoning=request.include_reasoning,
            cross_document=request.cross_document,
            extract_relationships=request.cross_document
        )
        
        # Get engine and perform search
        engine = get_deep_search_engine()
        
        # TODO: Load conversation history if conversation_id provided
        conversation_history = None
        
        result = await engine.search(
            user_id=user_id,
            query=request.query,
            config=config,
            conversation_history=conversation_history
        )
        
        # Convert to response model
        reasoning_steps = None
        if result.reasoning_chain:
            reasoning_steps = [
                ReasoningStep(
                    step=s["step"],
                    question=s["question"],
                    answer=s["answer"],
                    sources=s.get("sources", []),
                    confidence=s.get("confidence", 0.0)
                )
                for s in result.reasoning_chain
            ]
        
        doc_relationships = [
            DocumentRelationship(
                source=r.get("source", ""),
                target=r.get("target", ""),
                type=r.get("type", "relates-to"),
                strength=r.get("strength", 0.5),
                evidence=r.get("evidence", [])
            )
            for r in result.document_relationships
        ]
        
        sources = [
            Source(
                source=s.get("source", "document"),
                filename=s.get("filename", "Unknown"),
                page=s.get("page"),
                chunk_index=s.get("chunk_index"),
                relevance_score=s.get("relevance_score", 0.0),
                full_text=s.get("full_text"),
                matched_concepts=s.get("matched_concepts", [])
            )
            for s in result.sources
        ]
        
        return DeepSearchResponse(
            answer=result.answer,
            summary=result.summary,
            reasoning_chain=reasoning_steps,
            document_relationships=doc_relationships,
            confidence=result.confidence,
            completeness=result.completeness,
            processing_time_ms=result.processing_time_ms,
            sources=sources,
            follow_up_suggestions=result.follow_up_suggestions,
            related_concepts=result.related_concepts,
            query_analysis=result.query_analysis
        )
        
    except Exception as e:
        logger.error(f"Deep search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Deep search failed: {str(e)}"
        )


@router.post("/deep/reasoning", response_model=ReasoningResponse)
async def get_reasoning_steps(
    request: ReasoningRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get detailed reasoning steps for a query.
    
    Returns the full chain-of-thought reasoning process,
    including intermediate questions, answers, and confidence scores.
    """
    start_time = time.time()
    
    try:
        engine = get_deep_search_engine()
        
        config = DeepSearchConfig(
            depth=SearchDepth(request.depth) if request.depth in ["simple", "moderate", "deep"] else SearchDepth.MODERATE,
            include_reasoning=True
        )
        
        result = await engine.get_reasoning_steps(
            user_id=user_id,
            query=request.query,
            config=config
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        steps = [
            ReasoningStep(
                step=s["step"],
                question=s["question"],
                answer=s["answer"],
                confidence=s.get("confidence", 0.0)
            )
            for s in result.get("steps", [])
        ]
        
        return ReasoningResponse(
            query=result.get("query", request.query),
            steps=steps,
            final_answer=result.get("final_answer", ""),
            total_confidence=result.get("total_confidence", 0.0),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Reasoning error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning failed: {str(e)}"
        )


@router.get("/relationships")
async def get_relationships(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    concept: Optional[str] = Query(None, description="Filter by concept"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get document and concept relationships from the knowledge graph.
    
    Can filter by:
    - `document_id`: Get concepts and relationships for a specific document
    - `concept`: Get relationships involving a specific concept
    - Neither: Get all relationships for the user
    """
    try:
        engine = get_deep_search_engine()
        
        relationships = await engine.get_document_relationships(
            user_id=user_id,
            document_id=document_id,
            concept=concept,
            limit=limit
        )
        
        return {
            "relationships": relationships,
            "count": len(relationships),
            "filters": {
                "document_id": document_id,
                "concept": concept
            }
        }
        
    except Exception as e:
        logger.error(f"Relationships error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relationships: {str(e)}"
        )


@router.post("/cross-document", response_model=CrossDocumentResponse)
async def cross_document_search(
    request: CrossDocumentRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Search across multiple documents to find connections and relationships.
    
    This endpoint:
    - Extracts concepts from the query
    - Finds documents containing similar concepts
    - Calculates document similarities
    - Returns concept relationships across documents
    """
    try:
        engine = get_deep_search_engine()
        
        result = await engine.cross_document_search(
            user_id=user_id,
            query=request.query,
            document_ids=request.document_ids
        )
        
        return CrossDocumentResponse(
            query_concepts=result.get("query_concepts", []),
            related_documents=result.get("related_documents", []),
            document_similarities=result.get("document_similarities", []),
            concept_relationships=result.get("concept_relationships", [])
        )
        
    except Exception as e:
        logger.error(f"Cross-document search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cross-document search failed: {str(e)}"
        )


# ==================== Health Check ====================

@router.get("/deep/health")
async def deep_search_health():
    """Check deep search service health."""
    try:
        engine = get_deep_search_engine()
        
        # Check components
        has_analyzer = engine.query_analyzer is not None
        has_reasoner = engine.semantic_reasoner is not None
        has_graph = ping_neo4j()

        return {
            "status": "healthy" if all([has_analyzer, has_reasoner, has_graph]) else "degraded",
            "components": {
                "query_analyzer": "ok" if has_analyzer else "missing",
                "semantic_reasoner": "ok" if has_reasoner else "missing",
                "knowledge_graph": "ok" if has_graph else "unreachable (falls back to vector-only search)"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ==================== History Endpoints ====================

class HistoryItem(BaseModel):
    """A deep search history item."""
    search_id: str
    query: str
    depth: str
    answer: str
    confidence: float
    sources: List[dict] = []
    reasoning_chain: List[dict] = []
    relationships: List[dict] = []
    processing_time_ms: int = 0
    created_at: str


class HistoryResponse(BaseModel):
    """Response for history listing."""
    history: List[HistoryItem]
    total: int


@router.get("/deep/history", response_model=HistoryResponse)
async def get_search_history(
    limit: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get deep search history for the current user.
    """
    from db.mongo import get_deep_search_history
    
    try:
        history = get_deep_search_history(user_id, limit)
        
        items = [
            HistoryItem(
                search_id=h.get("search_id", ""),
                query=h.get("query", ""),
                depth=h.get("depth", "moderate"),
                answer=h.get("answer", ""),
                confidence=h.get("confidence", 0),
                sources=h.get("sources", []),
                reasoning_chain=h.get("reasoning_chain", []),
                relationships=h.get("relationships", []),
                processing_time_ms=h.get("processing_time_ms", 0),
                created_at=h.get("created_at").isoformat() if h.get("created_at") else ""
            )
            for h in history
        ]
        
        return HistoryResponse(
            history=items,
            total=len(items)
        )
        
    except Exception as e:
        logger.error(f"Failed to get search history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search history: {str(e)}"
        )


@router.get("/deep/history/{search_id}")
async def get_search_by_id(
    search_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get a specific deep search by ID.
    """
    from db.mongo import get_deep_search_by_id
    
    try:
        search = get_deep_search_by_id(user_id, search_id)
        
        if not search:
            raise HTTPException(
                status_code=404,
                detail="Search not found"
            )
        
        return {
            "search_id": search.get("search_id", ""),
            "query": search.get("query", ""),
            "depth": search.get("depth", "moderate"),
            "answer": search.get("answer", ""),
            "confidence": search.get("confidence", 0),
            "sources": search.get("sources", []),
            "reasoning_chain": search.get("reasoning_chain", []),
            "relationships": search.get("relationships", []),
            "processing_time_ms": search.get("processing_time_ms", 0),
            "created_at": search.get("created_at").isoformat() if search.get("created_at") else ""
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search: {str(e)}"
        )


@router.delete("/deep/history")
async def clear_search_history(
    user_id: str = Depends(get_current_user_id)
):
    """
    Clear all deep search history for the current user.
    """
    from db.mongo import delete_deep_search_history
    
    try:
        delete_deep_search_history(user_id)
        return {"message": "Search history cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear search history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear search history: {str(e)}"
        )
