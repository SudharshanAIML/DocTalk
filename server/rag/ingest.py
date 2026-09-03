from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag.vectorstore import get_user_vectorstore, save_user_vectorstore
from rag.graph_extractor import extract_page_graph_data
from db.mongo import insert_chunk
from rag import graph_store

def ingest_new_document(
    user_id: str,
    file_id: str,
    filename: str,
    extracted_pages: list[dict],
    file_type: str = "unknown"
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

    graph_store.upsert_document_node(user_id, file_id, filename, file_type)

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

        # Store chunk metadata in MongoDB, and mirror as Chunk nodes in Neo4j
        chunk_ids = []
        for i, doc in enumerate(docs):
            chunk_id = insert_chunk(
                user_id=user_id,
                file_id=file_id,
                page_number=page["page"],
                text_preview=doc.page_content,
                faiss_index_id=start_index + i
            )
            chunk_ids.append(chunk_id)
            graph_store.upsert_chunk_node(
                user_id=user_id,
                file_id=file_id,
                chunk_id=chunk_id,
                page=page["page"],
                faiss_index_id=start_index + i,
                text=doc.page_content
            )

        # Knowledge graph: extract concepts/relationships once per page (not per
        # chunk, to limit LLM calls), and link every chunk from this page to
        # whatever concepts were found on it.
        graph_data = extract_page_graph_data(page["text"])

        for concept in graph_data["concepts"]:
            concept_id = graph_store.upsert_concept(
                user_id=user_id,
                name=concept["name"],
                concept_type=concept.get("type", "term"),
                importance=concept.get("importance", 0.5)
            )
            for chunk_id in chunk_ids:
                graph_store.link_chunk_to_concept(chunk_id, concept_id)

        for rel in graph_data["relationships"]:
            graph_store.add_relationship(
                user_id=user_id,
                source_concept=rel["source"],
                target_concept=rel["target"],
                relationship_type=rel.get("type", "relates-to"),
                strength=rel.get("strength", 0.5),
                evidence=rel.get("evidence"),
                document_id=file_id
            )

    save_user_vectorstore(vectorstore, user_id)
