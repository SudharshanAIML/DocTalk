import os
import shutil
from langchain_community.vectorstores import FAISS
from rag.vectorstore import get_embeddings
from db.mongo import chunks_col

def rebuild_user_index(user_id: str):
    """
    Rebuild FAISS index from MongoDB chunks (after deletion).
    """
    chunks = list(chunks_col.find(
        {"user_id": user_id},
        {"_id": 0}
    ))

    if not chunks:
        shutil.rmtree(f"faiss_index/{user_id}", ignore_errors=True)
        return

    texts = [c["text_preview"] for c in chunks]
    metadatas = [
        {
            "file_id": c["file_id"],
            "page": c["page_number"]
        }
        for c in chunks
    ]

    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas)

    vectorstore.save_local(f"faiss_index/{user_id}")

