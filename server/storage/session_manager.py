"""
Session Manager for FAISS indices with automatic cleanup.
Manages in-memory cache and cloud storage synchronization.
Automatically deletes indices when session expires.
"""

import os
import io
import tempfile
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import logging
import pickle

from storage.cloud_storage import CloudFAISSStorage
from db.mongo import (
    documents_col,
    chunks_col,
    chat_history_col,
    faiss_metadata_col
)

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))  # Default 24 hours


class SessionManager:
    """
    Manages user sessions with FAISS indices.
    - Stores indices in memory for fast access
    - Syncs to cloud (MongoDB GridFS) periodically
    - Auto-cleans expired sessions from memory, cloud, and DB
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._cache: Dict[str, dict] = {}  # user_id -> {vectorstore, last_accessed, created_at}
        self._cache_lock = threading.Lock()
        self._embeddings = None
        self._cleanup_thread = None
        self._running = False
        
        # Start background cleanup
        self.start_cleanup_thread()
        
        logger.info("SessionManager initialized")
    
    def _get_embeddings(self):
        """Lazy load embeddings model."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings
    
    def start_cleanup_thread(self):
        """Start background thread for cleaning expired sessions."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self._cleanup_thread.start()
            logger.info("Cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop the cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
            logger.info("Cleanup thread stopped")
    
    def _cleanup_loop(self):
        """Background loop to clean expired sessions."""
        while self._running:
            try:
                self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
            
            # Check every 5 minutes
            time.sleep(300)
    
    def get_vectorstore(self, user_id: str) -> FAISS:
        """
        Get vectorstore for user. Loads from cloud if not in cache.
        Falls back to local storage for migration purposes.
        
        Args:
            user_id: User identifier
            
        Returns:
            FAISS vectorstore
        """
        with self._cache_lock:
            # Check cache first
            if user_id in self._cache:
                self._cache[user_id]["last_accessed"] = datetime.utcnow()
                vs = self._cache[user_id]["vectorstore"]
                logger.info(f"Loaded vectorstore from CACHE for user: {user_id}, vectors: {vs.index.ntotal}")
                return vs
            
            # Try to load from cloud
            cloud_data = CloudFAISSStorage.load_index_from_cloud(user_id)
            
            if cloud_data:
                vectorstore = self._load_vectorstore_from_bytes(cloud_data)
                if vectorstore:
                    self._cache[user_id] = {
                        "vectorstore": vectorstore,
                        "last_accessed": datetime.utcnow(),
                        "created_at": datetime.utcnow()
                    }
                    logger.info(f"Loaded vectorstore from CLOUD for user: {user_id}, vectors: {vectorstore.index.ntotal}")
                    return vectorstore
            
            # Fallback: Try to load from local storage (for migration)
            local_path = f"faiss_index/{user_id}"
            if os.path.exists(local_path):
                try:
                    vectorstore = FAISS.load_local(
                        local_path,
                        self._get_embeddings(),
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"Loaded vectorstore from LOCAL for user: {user_id}, vectors: {vectorstore.index.ntotal}")
                    
                    # Migrate to cloud
                    self._cache[user_id] = {
                        "vectorstore": vectorstore,
                        "last_accessed": datetime.utcnow(),
                        "created_at": datetime.utcnow()
                    }
                    
                    # Save to cloud for future use
                    index_bytes, pkl_bytes = self._vectorstore_to_bytes(vectorstore)
                    CloudFAISSStorage.save_index_to_cloud(user_id, index_bytes, pkl_bytes)
                    logger.info(f"Migrated local vectorstore to cloud for user: {user_id}")
                    
                    return vectorstore
                except Exception as e:
                    logger.error(f"Error loading from local storage: {e}")
            
            # Create new empty vectorstore
            vectorstore = FAISS.from_texts(["start"], self._get_embeddings())
            self._cache[user_id] = {
                "vectorstore": vectorstore,
                "last_accessed": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            logger.info(f"Created NEW vectorstore for user: {user_id}")
            return vectorstore
    
    def save_vectorstore(self, user_id: str, vectorstore: FAISS) -> bool:
        """
        Save vectorstore to cache and cloud.
        
        Args:
            user_id: User identifier
            vectorstore: FAISS vectorstore to save
            
        Returns:
            bool: Success status
        """
        try:
            # Update cache
            with self._cache_lock:
                self._cache[user_id] = {
                    "vectorstore": vectorstore,
                    "last_accessed": datetime.utcnow(),
                    "created_at": self._cache.get(user_id, {}).get("created_at", datetime.utcnow())
                }
            
            # Save to cloud
            index_bytes, pkl_bytes = self._vectorstore_to_bytes(vectorstore)
            success = CloudFAISSStorage.save_index_to_cloud(user_id, index_bytes, pkl_bytes)
            
            if success:
                logger.info(f"Saved vectorstore to cloud for user: {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving vectorstore: {e}")
            return False
    
    def _vectorstore_to_bytes(self, vectorstore: FAISS) -> tuple:
        """Convert vectorstore to bytes for cloud storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to temp directory
            vectorstore.save_local(tmpdir)
            
            # Read files as bytes
            with open(os.path.join(tmpdir, "index.faiss"), "rb") as f:
                index_bytes = f.read()
            
            with open(os.path.join(tmpdir, "index.pkl"), "rb") as f:
                pkl_bytes = f.read()
            
            return index_bytes, pkl_bytes
    
    def _load_vectorstore_from_bytes(self, data: tuple) -> Optional[FAISS]:
        """Load vectorstore from bytes."""
        try:
            index_bytes, pkl_bytes = data
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write bytes to temp files
                with open(os.path.join(tmpdir, "index.faiss"), "wb") as f:
                    f.write(index_bytes)
                
                with open(os.path.join(tmpdir, "index.pkl"), "wb") as f:
                    f.write(pkl_bytes)
                
                # Load vectorstore
                return FAISS.load_local(
                    tmpdir,
                    self._get_embeddings(),
                    allow_dangerous_deserialization=True
                )
        except Exception as e:
            logger.error(f"Error loading vectorstore from bytes: {e}")
            return None
    
    def refresh_session(self, user_id: str):
        """
        Refresh session timestamp (call on user activity).
        
        Args:
            user_id: User identifier
        """
        with self._cache_lock:
            if user_id in self._cache:
                self._cache[user_id]["last_accessed"] = datetime.utcnow()
        
        # Update cloud metadata
        from storage.cloud_storage import faiss_metadata_col
        faiss_metadata_col.update_one(
            {"user_id": user_id},
            {"$set": {"last_accessed": datetime.utcnow()}}
        )
    
    def is_session_expired(self, user_id: str) -> bool:
        """
        Check if session has expired.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if expired
        """
        metadata = CloudFAISSStorage.get_metadata(user_id)
        if not metadata:
            return True
        
        last_accessed = metadata.get("last_accessed")
        if not last_accessed:
            return True
        
        expiry_time = last_accessed + timedelta(hours=SESSION_TIMEOUT_HOURS)
        return datetime.utcnow() > expiry_time
    
    def cleanup_user_session(self, user_id: str) -> bool:
        """
        Clean up all data for a user session (manual cleanup).
        Deletes from: memory cache, cloud storage, and database.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Cleaning up session for user: {user_id}")
            
            # 1. Remove from memory cache
            with self._cache_lock:
                if user_id in self._cache:
                    del self._cache[user_id]
                    logger.info(f"Removed from memory cache: {user_id}")
            
            # 2. Delete from cloud storage (GridFS)
            CloudFAISSStorage.delete_from_cloud(user_id)
            logger.info(f"Removed from cloud storage: {user_id}")
            
            # 3. Delete from database (documents, chunks, chat history)
            documents_col.delete_many({"user_id": user_id})
            chunks_col.delete_many({"user_id": user_id})
            chat_history_col.delete_many({"user_id": user_id})
            logger.info(f"Removed from database: {user_id}")

            # 4. Delete from the knowledge graph
            from rag import graph_store
            graph_store.delete_user_graph(user_id)

            # 5. Delete uploaded files (if any local files exist)
            upload_dir = os.getenv("UPLOAD_DIR", "data/uploads")
            if os.path.exists(upload_dir):
                import glob
                for filepath in glob.glob(os.path.join(upload_dir, f"{user_id}_*")):
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed uploaded file: {filepath}")
                    except Exception as e:
                        logger.warning(f"Could not remove file {filepath}: {e}")
            
            logger.info(f"Session cleanup completed for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up session for {user_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """
        Evict expired sessions from the in-memory cache only.
        Called periodically by the background thread.

        This does NOT delete anything from cloud storage or the database —
        a user's documents/chunks/chat history must persist regardless of
        inactivity. Eviction just frees memory; get_vectorstore() reloads
        the index from GridFS on next access.
        """
        try:
            logger.info("Running expired session cache eviction...")

            # Clean memory cache
            with self._cache_lock:
                expired_in_cache = []
                for user_id, session in self._cache.items():
                    last_accessed = session.get("last_accessed", datetime.utcnow())
                    if datetime.utcnow() > last_accessed + timedelta(hours=SESSION_TIMEOUT_HOURS):
                        expired_in_cache.append(user_id)
                
                for user_id in expired_in_cache:
                    del self._cache[user_id]
                    logger.info(f"Removed expired session from cache: {user_id}")

            logger.info(f"Cache eviction completed. Evicted {len(expired_in_cache)} expired sessions.")

        except Exception as e:
            logger.error(f"Error in cleanup_expired_sessions: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the session cache."""
        with self._cache_lock:
            return {
                "cached_sessions": len(self._cache),
                "user_ids": list(self._cache.keys())
            }


# Global session manager instance
session_manager = SessionManager()


# Convenience functions
def get_user_vectorstore(user_id: str) -> FAISS:
    """Get vectorstore for user."""
    return session_manager.get_vectorstore(user_id)


def save_user_vectorstore(vectorstore: FAISS, user_id: str) -> bool:
    """Save vectorstore for user."""
    return session_manager.save_vectorstore(user_id, vectorstore)


def refresh_user_session(user_id: str):
    """Refresh user session timestamp."""
    session_manager.refresh_session(user_id)


def cleanup_user_session(user_id: str) -> bool:
    """Clean up user session completely."""
    return session_manager.cleanup_user_session(user_id)


def is_session_expired(user_id: str) -> bool:
    """Check if user session is expired."""
    return session_manager.is_session_expired(user_id)
