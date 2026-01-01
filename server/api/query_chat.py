from fastapi import APIRouter, Depends
from rag.memory_chain import get_conversational_rag_chain

router = APIRouter(prefix="/chat", tags=["Chat"])

def get_current_user_id():
    return "test-user-id"

# Store chain per user session (in-memory)
user_chains = {}

@router.post("/")
def chat_query(
    question: str,
    user_id: str = Depends(get_current_user_id)
):
    if user_id not in user_chains:
        user_chains[user_id] = get_conversational_rag_chain(user_id)

    chain = user_chains[user_id]
    response = chain({"question": question})

    return {
        "answer": response["answer"],
        "sources": [
            {
                "file": doc.metadata.get("filename"),
                "page": doc.metadata.get("page")
            }
            for doc in response["source_documents"]
        ]
    }
