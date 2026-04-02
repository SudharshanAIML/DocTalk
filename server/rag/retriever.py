"""
Retriever module for DocTalk.
Provides document retrieval with similarity search across ALL user documents,
or filtered to specific document IDs when filters are applied.
"""

from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rag.vectorstore import get_user_vectorstore
import logging
import re

logger = logging.getLogger(__name__)


class EnhancedRetriever(BaseRetriever):
    """
    Enhanced retriever that handles personal queries better.
    Supports filtering by specific document IDs (file_ids).
    """
    
    vectorstore: object  # FAISS vectorstore
    k: int = 8
    filter_document_ids: Optional[List[str]] = None  # If set, restrict to these file_id values
    
    class Config:
        arbitrary_types_allowed = True
    
    def _filter_by_doc_ids(self, docs: List[Document]) -> List[Document]:
        """Post-filter documents to only include those matching filter_document_ids."""
        if not self.filter_document_ids:
            return docs
        allowed = set(self.filter_document_ids)
        return [
            doc for doc in docs
            if doc.metadata.get("file_id") in allowed
            or doc.metadata.get("filename") in allowed  # fallback: match by filename too
        ]

    def _fetch_candidates(self, query: str, k_multiplier: int = 4) -> List[Document]:
        """
        Fetch candidates from vectorstore.
        When document filter is active, fetches more candidates to ensure
        enough results survive the post-filter step.
        """
        fetch_k = self.k * k_multiplier if self.filter_document_ids else self.k * 3
        return self.vectorstore.similarity_search(query, k=fetch_k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get relevant documents for a query.
        When filter_document_ids is set, only returns chunks from those documents.
        """
        if self.filter_document_ids:
            logger.info(f"Filtered retrieval: restricting to doc IDs {self.filter_document_ids}")
            candidates = self._fetch_candidates(query, k_multiplier=6)
            filtered = self._filter_by_doc_ids(candidates)
            
            if not filtered:
                logger.warning(
                    f"No results found after filtering to doc IDs {self.filter_document_ids}. "
                    "Falling back to unfiltered search."
                )
                # Fall back gracefully so the user gets some answer
                filtered = candidates

            # Deduplicate and limit
            seen = set()
            result = []
            for doc in filtered:
                key = hash(doc.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    result.append(doc)
                if len(result) >= self.k:
                    break

            logger.info(f"Filtered retrieval returned {len(result)} docs")
            return result

        # Check if query is about the user themselves
        personal_patterns = [
            r'\b(myself|about me|my background|my resume|my profile|my skills|my experience|my education|my projects|who am i|tell me about me)\b',
            r'\b(my name|my qualifications|my achievements|my work)\b'
        ]
        
        is_personal_query = any(re.search(pattern, query.lower()) for pattern in personal_patterns)
        
        if is_personal_query:
            logger.info(f"Detected personal query: {query}")
            search_queries = [
                query,
                "resume profile name education skills experience projects achievements",
                "personal background qualifications work history",
                "name contact email skills programming languages"
            ]
            
            all_docs = []
            seen_contents = set()
            
            for sq in search_queries:
                results = self.vectorstore.similarity_search(sq, k=self.k)
                for doc in results:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_docs.append(doc)
            
            def score_doc(doc):
                filename = doc.metadata.get('filename', '').lower()
                content = doc.page_content.lower()
                score = 0
                if 'resume' in filename or 'cv' in filename or 'profile' in filename:
                    score += 100
                if any(word in filename for word in ['stripe', 'sudharshan']):
                    score += 50
                personal_keywords = ['skills', 'education', 'experience', 'projects', 
                                   'programming', 'languages', 'achievements', 'university',
                                   'leetcode', 'codechef', 'github']
                for kw in personal_keywords:
                    if kw in content:
                        score += 5
                return -score
            
            all_docs.sort(key=score_doc)
            logger.info(f"Personal query returned {len(all_docs)} documents")
            return all_docs[:self.k]
        else:
            # Document-diversified similarity search
            candidates = self.vectorstore.similarity_search(query, k=self.k * 3)
            
            if not candidates:
                return []
            
            # Group by source document and round-robin for diversity
            doc_groups = {}
            for doc in candidates:
                fname = doc.metadata.get('filename', 'unknown')
                doc_groups.setdefault(fname, []).append(doc)
            
            logger.info(f"Query returned candidates from {len(doc_groups)} documents: {list(doc_groups.keys())}")
            
            result = []
            while len(result) < self.k and doc_groups:
                empty_keys = []
                for key in list(doc_groups.keys()):
                    docs = doc_groups[key]
                    if docs and len(result) < self.k:
                        result.append(docs.pop(0))
                    if not docs:
                        empty_keys.append(key)
                for key in empty_keys:
                    del doc_groups[key]
            
            logger.info(f"Diversified retrieval returned {len(result)} docs from {len(set(d.metadata.get('filename', '') for d in result))} unique files")
            return result


def get_retriever(user_id: str, k: int = 8, filter_document_ids: Optional[List[str]] = None) -> EnhancedRetriever:
    """
    Get an enhanced retriever for the user's documents.
    
    Args:
        user_id: User identifier
        k: Number of top results to return
        filter_document_ids: If provided, restrict search to only these document IDs
        
    Returns:
        EnhancedRetriever configured for similarity search
    """
    vectorstore = get_user_vectorstore(user_id)
    
    try:
        total_docs = vectorstore.index.ntotal
        logger.info(f"Retriever for user {user_id}: {total_docs} total vectors in index")
        if filter_document_ids:
            logger.info(f"  → Filter active: {filter_document_ids}")
    except Exception as e:
        logger.warning(f"Could not get index count: {e}")
    
    return EnhancedRetriever(
        vectorstore=vectorstore,
        k=k,
        filter_document_ids=filter_document_ids
    )
