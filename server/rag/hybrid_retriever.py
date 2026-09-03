"""
Hybrid retriever: fuses FAISS vector similarity search with Neo4j knowledge
graph traversal, via Reciprocal Rank Fusion. Used by both the main chat chain
(rag/memory_chain.py) and Deep Search (rag/deep_search.py).

Falls back to vector-only results whenever the graph has nothing to add
(no concepts matched, or Neo4j unreachable) — graph_store's read functions
already degrade to empty results on any Neo4j error, so this retriever never
needs to special-case connectivity failures itself.
"""

import logging
from typing import List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag.retriever import EnhancedRetriever
from rag.vectorstore import get_user_vectorstore
from rag.query_analyzer import analyze_query
from rag.graph_store import find_chunks_by_concepts

logger = logging.getLogger(__name__)

RRF_K = 60


class HybridGraphRetriever(BaseRetriever):
    """Vector search + knowledge graph traversal, fused via Reciprocal Rank Fusion."""

    vectorstore: object
    user_id: str
    k: int = 8
    filter_document_ids: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vector_retriever = EnhancedRetriever(
            vectorstore=self.vectorstore, k=self.k, filter_document_ids=self.filter_document_ids
        )
        vector_docs = vector_retriever.invoke(query)

        # Document-filtered queries already work well on vector search alone;
        # graph traversal isn't filter-scoped, so skip it in that case.
        if self.filter_document_ids:
            return vector_docs

        graph_docs = self._graph_documents(query)
        if not graph_docs:
            return vector_docs

        return self._reciprocal_rank_fusion(vector_docs, graph_docs)[: self.k]

    def _graph_documents(self, query: str) -> List[Document]:
        try:
            concepts = analyze_query(query).concepts
        except Exception as e:
            logger.warning(f"Query concept extraction failed: {e}")
            return []

        if not concepts:
            return []

        try:
            chunks = find_chunks_by_concepts(self.user_id, concepts, limit=self.k * 2)
        except Exception as e:
            logger.warning(f"Graph chunk lookup failed: {e}")
            return []

        return [
            Document(
                page_content=chunk["content"],
                metadata={
                    "filename": chunk.get("filename"),
                    "page": chunk.get("page"),
                    "file_id": chunk.get("file_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "matched_concepts": chunk.get("matched_concepts", []),
                    "source_type": "graph",
                },
            )
            for chunk in chunks
            if chunk.get("content")
        ]

    def _reciprocal_rank_fusion(
        self, vector_docs: List[Document], graph_docs: List[Document]
    ) -> List[Document]:
        scores = {}
        merged = {}

        for rank, doc in enumerate(vector_docs):
            key = hash(doc.page_content[:200])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            merged.setdefault(key, doc)

        for rank, doc in enumerate(graph_docs):
            key = hash(doc.page_content[:200])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key in merged:
                existing = merged[key]
                existing_concepts = set(existing.metadata.get("matched_concepts") or [])
                existing_concepts.update(doc.metadata.get("matched_concepts") or [])
                existing.metadata["matched_concepts"] = list(existing_concepts)
                existing.metadata["source_type"] = "both"
            else:
                merged[key] = doc

        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [merged[key] for key in ranked_keys]


def get_hybrid_retriever(
    user_id: str, k: int = 8, filter_document_ids: Optional[List[str]] = None
) -> HybridGraphRetriever:
    """Get a hybrid (vector + graph) retriever for the user's documents."""
    vectorstore = get_user_vectorstore(user_id)
    return HybridGraphRetriever(
        vectorstore=vectorstore, user_id=user_id, k=k, filter_document_ids=filter_document_ids
    )
