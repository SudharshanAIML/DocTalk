from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
import os
import shutil
import uuid

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from db.mongo import (
    insert_document,
    get_user_documents,
    delete_document,
    delete_chunks_by_file
)
from rag.ingest import ingest_new_document
from rag.reindex import rebuild_user_faiss_index
from rag.parallel_ingest import parallel_ingest_document


# Router

router = APIRouter(prefix="/documents", tags=["Documents"])



# AUTH PLACEHOLDER (replace with JWT later)

def get_current_user_id():
    """
    TEMP: Replace with JWT-based auth later
    """
    return "test-user-id"



# UTILS

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



# LOAD DOCUMENT USING LANGCHAIN

def load_document_with_langchain(file_path: str, filename: str):
    """
    Uses LangChain loaders to extract text.
    Returns list of:
    [{"page": int, "text": str}]
    """

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        file_type = "pdf"

        pages = [
            {
                "page": doc.metadata.get("page", i) + 1,
                "text": doc.page_content
            }
            for i, doc in enumerate(docs)
        ]

    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        file_type = "docx"

        # DOCX usually has no pages
        pages = [
            {
                "page": 1,
                "text": docs[0].page_content
            }
        ]

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    return pages, file_type



# UPLOAD DOCUMENT

@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user_id)
):
    uploaded_files = []

    for file in files:
        if not file.filename.endswith((".pdf", ".docx")):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and DOCX files are supported"
            )

        # Save file
        temp_file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{temp_file_id}_{file.filename}")

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 🔥 Load using LangChain
        pages, file_type = load_document_with_langchain(file_path, file.filename)

        # Save document metadata
        doc = insert_document(
            user_id=user_id,
            filename=file.filename,
            file_type=file_type,
            num_pages=len(pages)
        )

        # 🔥 Incremental ingestion (LangChain + FAISS)
        parallel_ingest_document(
            user_id=user_id,
            file_id=doc["file_id"],
            filename=file.filename,
            extracted_pages=pages
        )

        uploaded_files.append({
            "file_id": doc["file_id"],
            "filename": file.filename
        })

    return {
        "message": "Files uploaded and indexed successfully",
        "files": uploaded_files
    }



# LIST USER DOCUMENTS

@router.get("/")
def list_documents(user_id: str = Depends(get_current_user_id)):
    return get_user_documents(user_id)



# DELETE DOCUMENT (WITH RE-INDEX)

@router.delete("/{file_id}")
def delete_user_document(
    file_id: str,
    user_id: str = Depends(get_current_user_id)
):
    delete_document(user_id, file_id)
    delete_chunks_by_file(user_id, file_id)

    # 🔥 Rebuild FAISS index (only on delete)
    rebuild_user_faiss_index(user_id)

    return {"message": "Document deleted and vector index rebuilt"}
