from concurrent.futures import ThreadPoolExecutor
from rag.ingest import ingest_new_document

def parallel_ingest_document(
    user_id: str,
    file_id: str,
    filename: str,
    extracted_pages: list,
    max_workers: int = 4
):
    """
    Wrapper for ingestion. 
    NOTE: Parallel ingestion is disabled because FAISS file persistence is not thread-safe.
    Concurrent writes to the same 'faiss_index/{user_id}' folder result in data loss.
    """
    ingest_new_document(user_id, file_id, filename, extracted_pages)
