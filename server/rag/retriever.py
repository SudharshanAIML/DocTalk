from rag.vectorstore import get_user_vectorstore

def get_retriever(user_id: str, k: int = 3):
    vectorstore = get_user_vectorstore(user_id)
    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
