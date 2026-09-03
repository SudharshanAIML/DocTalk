"""
Deep Search Module for DocTalk.

Main orchestrator for the deep search pipeline:
1. Query Understanding
2. Multi-layer Retrieval
3. Deep Analysis with Reasoning
4. Intelligent Ranking
5. Result Synthesis

Provides comprehensive, reasoned answers with citations.
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from dotenv import load_dotenv

from rag.query_analyzer import (
    QueryAnalyzer, AnalyzedQuery, QueryComplexity, QueryType,
    get_query_analyzer, analyze_query
)
from rag.semantic_reasoner import (
    SemanticReasoner, ReasoningChain, ReasoningDepth,
    get_semantic_reasoner
)
from rag import graph_store
from rag.graph_store import GraphQuery
from rag.result_synthesizer import (
    ResultRanker, ResultSynthesizer, RankedResult, SynthesizedAnswer,
    rank_results, synthesize_answer
)
from rag.hybrid_retriever import get_hybrid_retriever
from db.mongo import save_deep_search
from rag.vectorstore import get_user_vectorstore

load_dotenv()
logger = logging.getLogger(__name__)


class SearchDepth(Enum):
    """Search depth levels matching API."""
    SIMPLE = "simple"      # Fast, basic search
    MODERATE = "moderate"  # Balanced search with some reasoning
    DEEP = "deep"          # Thorough search with full reasoning


@dataclass
class DeepSearchConfig:
    """Configuration for deep search."""
    # Search settings
    depth: SearchDepth = SearchDepth.MODERATE
    max_chunks: int = 12
    include_reasoning: bool = True
    cross_document: bool = True
    
    # Token/context limits
    max_context_tokens: int = 20000
    max_reasoning_steps: int = 5
    
    # Timeouts
    timeout_seconds: int = 30
    
    # Caching
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    # Feature flags
    build_knowledge_graph: bool = True
    extract_relationships: bool = True
    find_implicit_info: bool = True
    generate_suggestions: bool = True


@dataclass
class DeepSearchResult:
    """Complete result from deep search."""
    # Main answer
    answer: str
    summary: str
    
    # Reasoning information
    reasoning_chain: Optional[List[Dict]] = None
    reasoning_quality: str = "medium"
    
    # Document relationships
    document_relationships: List[Dict] = field(default_factory=list)
    
    # Sources and citations
    sources: List[Dict] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    
    # Scores
    confidence: float = 0.0
    completeness: float = 0.0
    
    # Metadata
    processing_time_ms: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)
    query_analysis: Dict = field(default_factory=dict)
    
    # Suggestions
    follow_up_suggestions: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)


class DeepSearchEngine:
    """
    Main engine for deep search functionality.
    Orchestrates the entire deep search pipeline.
    """
    
    def __init__(
        self,
        query_analyzer: Optional[QueryAnalyzer] = None,
        semantic_reasoner: Optional[SemanticReasoner] = None,
        result_ranker: Optional[ResultRanker] = None,
        result_synthesizer: Optional[ResultSynthesizer] = None
    ):
        """
        Initialize the deep search engine.

        Args:
            query_analyzer: Query analyzer instance
            semantic_reasoner: Semantic reasoner instance
            result_ranker: Result ranker instance
            result_synthesizer: Result synthesizer instance
        """
        self.query_analyzer = query_analyzer or get_query_analyzer()
        self.semantic_reasoner = semantic_reasoner or get_semantic_reasoner()
        self.result_ranker = result_ranker or ResultRanker()
        self.result_synthesizer = result_synthesizer or ResultSynthesizer(self.result_ranker)
        
        # Cache for reasoning results
        self._reasoning_cache: Dict[str, Dict] = {}
    
    async def search(
        self,
        user_id: str,
        query: str,
        config: Optional[DeepSearchConfig] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> DeepSearchResult:
        """
        Perform a deep search.
        
        Args:
            user_id: User identifier
            query: Search query
            config: Search configuration
            conversation_history: Previous conversation for context
            
        Returns:
            DeepSearchResult with comprehensive answer
        """
        start_time = time.time()
        config = config or DeepSearchConfig()
        
        logger.info(f"Starting deep search for user {user_id}: {query[:100]}...")
        
        try:
            # ========== STAGE 1: Query Understanding ==========
            analyzed_query = self.query_analyzer.analyze(query, {
                "conversation_history": conversation_history
            })
            logger.info(f"Query analyzed: type={analyzed_query.query_type.value}, "
                       f"complexity={analyzed_query.complexity.value}")
            
            # ========== STAGE 2: Initial Retrieval ==========
            context_chunks = await self._retrieve_context(
                user_id=user_id,
                analyzed_query=analyzed_query,
                config=config
            )
            logger.info(f"Retrieved {len(context_chunks)} context chunks")
            
            if not context_chunks:
                return DeepSearchResult(
                    answer="No relevant information found in your uploaded documents.",
                    summary="No results found.",
                    confidence=0.0,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    query_analysis=self._serialize_query_analysis(analyzed_query)
                )
            
            # ========== STAGE 3: Deep Analysis (if needed) ==========
            reasoning_chain = None
            if config.include_reasoning and analyzed_query.requires_reasoning:
                reasoning_chain = await self._perform_reasoning(
                    query=query,
                    analyzed_query=analyzed_query,
                    context=context_chunks,
                    config=config,
                    conversation_history=conversation_history
                )
                logger.info(f"Reasoning complete: {len(reasoning_chain.steps)} steps")
            
            # ========== STAGE 4: Intelligent Ranking ==========
            reasoning_context = None
            if reasoning_chain:
                reasoning_context = {
                    "sub_questions": [{"question": sq.question} for sq in analyzed_query.sub_questions],
                    "reasoning_steps": [{"answer": s.answer} for s in reasoning_chain.steps]
                }
            
            ranked_results = self.result_ranker.rank_results(
                results=context_chunks,
                query=query,
                query_concepts=analyzed_query.concepts,
                reasoning_context=reasoning_context
            )
            logger.info(f"Ranked {len(ranked_results)} results")
            
            # ========== STAGE 5: Result Synthesis ==========
            reasoning_dict = self._reasoning_chain_to_dict(reasoning_chain) if reasoning_chain else None
            synthesized = self.result_synthesizer.synthesize(
                query=query,
                ranked_results=ranked_results,
                reasoning_chain=reasoning_dict,
                include_suggestions=config.generate_suggestions
            )
            
            # ========== Build Document Relationships ==========
            doc_relationships = []
            if config.extract_relationships and config.cross_document:
                doc_relationships = await self._extract_document_relationships(
                    user_id=user_id,
                    context=context_chunks,
                    concepts=analyzed_query.concepts
                )
            
            # ========== Build Final Result ==========
            processing_time = int((time.time() - start_time) * 1000)
            
            token_usage = {}
            if reasoning_chain:
                token_usage = reasoning_chain.token_usage
            
            # Format results
            formatted_sources = self._format_sources(ranked_results[:8])
            formatted_reasoning = self._format_reasoning_steps(reasoning_chain)
            
            result = DeepSearchResult(
                answer=synthesized.answer,
                summary=synthesized.summary,
                reasoning_chain=formatted_reasoning,
                reasoning_quality=synthesized.reasoning_quality,
                document_relationships=doc_relationships,
                sources=formatted_sources,
                citations=[self._citation_to_dict(c) for c in synthesized.citations],
                confidence=synthesized.confidence,
                completeness=synthesized.completeness,
                processing_time_ms=processing_time,
                token_usage=token_usage,
                query_analysis=self._serialize_query_analysis(analyzed_query),
                follow_up_suggestions=synthesized.suggestions,
                related_concepts=synthesized.key_concepts
            )
            
            # ========== Save to History ==========
            try:
                save_deep_search(
                    user_id=user_id,
                    query=query,
                    depth=config.depth.value,
                    result={
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "sources": result.sources
                    },
                    reasoning_chain=result.reasoning_chain or [],
                    relationships=result.document_relationships,
                    processing_time_ms=processing_time
                )
            except Exception as save_err:
                logger.warning(f"Failed to save deep search to history: {save_err}")
            
            return result
            
        except Exception as e:
            logger.error(f"Deep search error: {e}", exc_info=True)
            processing_time = int((time.time() - start_time) * 1000)
            
            return DeepSearchResult(
                answer=f"An error occurred during deep search: {str(e)}",
                summary="Error during search",
                confidence=0.0,
                processing_time_ms=processing_time,
                query_analysis={"error": str(e)}
            )
    
    async def _retrieve_context(
        self,
        user_id: str,
        analyzed_query: AnalyzedQuery,
        config: DeepSearchConfig
    ) -> List[Dict]:
        """Retrieve context using multiple strategies."""
        all_chunks = []
        seen_contents = set()
        
        # Determine how many chunks to retrieve based on depth
        k_multiplier = {
            SearchDepth.SIMPLE: 1,
            SearchDepth.MODERATE: 2,
            SearchDepth.DEEP: 3
        }.get(config.depth, 2)
        
        base_k = config.max_chunks
        total_k = base_k * k_multiplier
        
        # Get retriever (hybrid: vector + knowledge graph)
        retriever = get_hybrid_retriever(user_id, k=total_k)
        
        # Main query retrieval
        main_results = retriever.invoke(analyzed_query.original_query)
        
        for doc in main_results:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_chunks.append(doc)
        
        # Sub-question retrieval for complex queries
        if config.depth in [SearchDepth.MODERATE, SearchDepth.DEEP]:
            for sq in analyzed_query.sub_questions[:3]:
                question = sq.question if hasattr(sq, 'question') else str(sq)
                sq_results = retriever.invoke(question)
                
                for doc in sq_results:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_chunks.append(doc)
        
        # Concept-based retrieval for deep searches
        if config.depth == SearchDepth.DEEP and analyzed_query.concepts:
            concept_query = " ".join(analyzed_query.concepts[:5])
            concept_results = retriever.invoke(concept_query)
            
            for doc in concept_results:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_chunks.append(doc)
        
        # Limit to max chunks
        return all_chunks[:config.max_chunks * 2]
    
    async def _perform_reasoning(
        self,
        query: str,
        analyzed_query: AnalyzedQuery,
        context: List[Dict],
        config: DeepSearchConfig,
        conversation_history: Optional[List[Dict]]
    ) -> ReasoningChain:
        """Perform semantic reasoning on the query."""
        # Map search depth to reasoning depth
        reasoning_depth = {
            SearchDepth.SIMPLE: ReasoningDepth.QUICK,
            SearchDepth.MODERATE: ReasoningDepth.STANDARD,
            SearchDepth.DEEP: ReasoningDepth.DEEP
        }.get(config.depth, ReasoningDepth.STANDARD)
        
        # Prepare sub-questions
        sub_questions = None
        if analyzed_query.sub_questions:
            sub_questions = [
                {"question": sq.question, "purpose": sq.purpose}
                for sq in analyzed_query.sub_questions[:config.max_reasoning_steps]
            ]
        
        # Perform reasoning
        return await self.semantic_reasoner.reason(
            query=query,
            context=context,
            sub_questions=sub_questions,
            depth=reasoning_depth,
            conversation_history=conversation_history
        )
    
    async def _extract_document_relationships(
        self,
        user_id: str,
        context: List[Dict],
        concepts: List[str]
    ) -> List[Dict]:
        """Extract relationships between documents."""
        relationships = []
        
        try:
            # Get relationships from semantic reasoner
            rels = await self.semantic_reasoner.extract_relationships(
                context=context,
                focus_concepts=concepts[:5] if concepts else None
            )
            
            for rel in rels:
                relationships.append({
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.relationship_type,
                    "strength": rel.strength,
                    "evidence": rel.evidence[:1] if rel.evidence else []
                })
            
            # Store in knowledge graph for future use
            for rel in rels:
                graph_store.add_relationship(
                    user_id=user_id,
                    source_concept=rel.source,
                    target_concept=rel.target,
                    relationship_type=rel.relationship_type,
                    strength=rel.strength,
                    evidence=rel.evidence[0] if rel.evidence else None
                )
                
        except Exception as e:
            logger.warning(f"Failed to extract relationships: {e}")
        
        return relationships
    
    def _reasoning_chain_to_dict(self, chain: ReasoningChain) -> Dict:
        """Convert ReasoningChain to dictionary."""
        return {
            "query": chain.query,
            "steps": [
                {
                    "step": s.step_number,
                    "question": s.question,
                    "reasoning": s.reasoning,
                    "answer": s.answer,
                    "confidence": s.confidence
                }
                for s in chain.steps
            ],
            "final_answer": chain.final_answer,
            "total_confidence": chain.total_confidence
        }
    
    def _format_reasoning_steps(
        self,
        chain: Optional[ReasoningChain]
    ) -> Optional[List[Dict]]:
        """Format reasoning steps for API response."""
        if not chain:
            return None
        
        return [
            {
                "step": step.step_number,
                "question": step.question,
                "answer": step.answer,
                "sources": step.sources,
                "confidence": step.confidence
            }
            for step in chain.steps
        ]
    
    def _format_sources(self, ranked_results: List[RankedResult]) -> List[Dict]:
        """Format ranked results as sources."""
        sources = []
        for result in ranked_results:
            sources.append({
                "source": "document",
                "filename": result.source,
                "page": result.page,
                "chunk_index": result.chunk_index,
                "relevance_score": result.relevance_score,
                "full_text": result.content[:500] if result.content else None,
                "matched_concepts": result.matched_concepts
            })
        return sources
    
    def _citation_to_dict(self, citation) -> Dict:
        """Convert Citation to dictionary."""
        return {
            "id": citation.citation_id,
            "source": citation.source,
            "page": citation.page,
            "quote": citation.quote,
            "relevance": citation.relevance
        }
    
    def _serialize_query_analysis(self, analyzed: AnalyzedQuery) -> Dict:
        """Serialize analyzed query for response."""
        return {
            "query_type": analyzed.query_type.value,
            "complexity": analyzed.complexity.value,
            "entities": [
                {"text": e.text, "type": e.entity_type}
                for e in analyzed.entities
            ],
            "concepts": analyzed.concepts,
            "sub_questions": [
                {"question": sq.question, "purpose": sq.purpose}
                for sq in analyzed.sub_questions
            ],
            "requires_reasoning": analyzed.requires_reasoning,
            "requires_cross_document": analyzed.requires_cross_document,
            "estimated_steps": analyzed.estimated_steps,
            "confidence": analyzed.confidence
        }
    
    # ==================== Additional Methods ====================
    
    async def get_reasoning_steps(
        self,
        user_id: str,
        query: str,
        config: Optional[DeepSearchConfig] = None
    ) -> Dict:
        """Get detailed reasoning steps for a query."""
        config = config or DeepSearchConfig(include_reasoning=True)
        
        # Analyze query
        analyzed = self.query_analyzer.analyze(query)
        
        # Get context
        context = await self._retrieve_context(user_id, analyzed, config)
        
        # Perform reasoning
        chain = await self._perform_reasoning(
            query=query,
            analyzed_query=analyzed,
            context=context,
            config=config,
            conversation_history=None
        )
        
        return self._reasoning_chain_to_dict(chain)
    
    async def get_document_relationships(
        self,
        user_id: str,
        document_id: Optional[str] = None,
        concept: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get document/concept relationships from knowledge graph."""
        if concept:
            # Get relationships for a concept
            rels = graph_store.get_relationships(
                user_id=user_id,
                concept=concept,
                limit=limit
            )
            return [
                {
                    "source": r.source_concept,
                    "target": r.target_concept,
                    "type": r.relationship_type,
                    "strength": r.strength
                }
                for r in rels
            ]
        elif document_id:
            # Get concepts for a document
            concepts = graph_store.get_document_concepts(user_id, document_id)
            return [
                {
                    "concept": c.name,
                    "type": c.concept_type,
                    "importance": c.importance
                }
                for c in concepts
            ]
        else:
            # Get all relationships
            rels = graph_store.get_relationships(user_id, limit=limit)
            return [
                {
                    "source": r.source_concept,
                    "target": r.target_concept,
                    "type": r.relationship_type,
                    "strength": r.strength
                }
                for r in rels
            ]
    
    async def cross_document_search(
        self,
        user_id: str,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict:
        """Search across multiple documents for connections."""
        # Analyze query
        analyzed = self.query_analyzer.analyze(query)
        
        # Find similar documents based on concepts
        similar_docs = []
        for concept in analyzed.concepts[:5]:
            c = graph_store.get_concept(user_id, concept)
            if c and c.document_ids:
                similar_docs.extend(c.document_ids)
        
        # Get unique documents
        unique_docs = list(set(similar_docs))
        
        # Find document similarities
        doc_similarities = []
        if document_ids and len(document_ids) > 0:
            for doc_id in document_ids:
                sims = graph_store.find_similar_documents(user_id, doc_id)
                for sim_id, score in sims:
                    doc_similarities.append({
                        "document_1": doc_id,
                        "document_2": sim_id,
                        "similarity": score
                    })
        
        # Query the graph
        graph_result = graph_store.query_graph(
            user_id=user_id,
            query=GraphQuery(concepts=analyzed.concepts[:5])
        )
        
        return {
            "query_concepts": analyzed.concepts,
            "related_documents": unique_docs,
            "document_similarities": doc_similarities,
            "concept_relationships": [
                {
                    "source": r.source_concept,
                    "target": r.target_concept,
                    "type": r.relationship_type,
                    "strength": r.strength
                }
                for r in graph_result.relationships
            ]
        }


# Global instance
_deep_search_engine = None


def get_deep_search_engine() -> DeepSearchEngine:
    """Get the global deep search engine instance."""
    global _deep_search_engine
    if _deep_search_engine is None:
        _deep_search_engine = DeepSearchEngine()
    return _deep_search_engine


async def deep_search(
    user_id: str,
    query: str,
    depth: str = "moderate",
    include_reasoning: bool = True,
    conversation_history: Optional[List[Dict]] = None
) -> DeepSearchResult:
    """
    Convenience function for deep search.
    
    Args:
        user_id: User identifier
        query: Search query
        depth: Search depth (simple, moderate, deep)
        include_reasoning: Whether to include reasoning steps
        conversation_history: Previous conversation
        
    Returns:
        DeepSearchResult
    """
    engine = get_deep_search_engine()
    
    config = DeepSearchConfig(
        depth=SearchDepth(depth),
        include_reasoning=include_reasoning
    )
    
    return await engine.search(
        user_id=user_id,
        query=query,
        config=config,
        conversation_history=conversation_history
    )
