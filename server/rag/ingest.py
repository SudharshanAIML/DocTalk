from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag.vectorstore import get_user_vectorstore, save_user_vectorstore
from db.mongo import insert_chunk

def ingest_new_document(
    user_id: str,
    file_id: str,
    filename: str,
    extracted_pages: list[dict]
):
    """
    extracted_pages = [
        {"page": 1, "text": "..."},
        {"page": 2, "text": "..."}
    ]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    vectorstore = get_user_vectorstore(user_id)

    for page in extracted_pages:
        docs = splitter.create_documents(
            [page["text"]],
            metadatas=[{
                "user_id": user_id,
                "file_id": file_id,
                "filename": filename,
                "page": page["page"]
            }]
        )

        # Get the current count before adding new docs
        start_index = vectorstore.index.ntotal
        
        # 🔥 ONLY new docs are added
        vectorstore.add_documents(docs)

        # Store chunk metadata in MongoDB
        for i, doc in enumerate(docs):
            insert_chunk(
                user_id=user_id,
                file_id=file_id,
                page_number=page["page"],
                text_preview=doc.page_content,
                faiss_index_id=start_index + i
            )

    save_user_vectorstore(vectorstore, user_id)
