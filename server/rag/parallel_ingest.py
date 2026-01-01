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
    Ingest pages in parallel (safe for FAISS append).
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for page in extracted_pages:
            executor.submit(
                ingest_new_document,
                user_id,
                file_id,
                filename,
                [page]  # single-page ingestion
            )
