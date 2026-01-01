from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from rag.streaming_chain import get_streaming_rag_chain

router = APIRouter(prefix="/query", tags=["Query"])

def get_current_user_id():
    return "test-user-id"

@router.post("/stream")
def query_stream(
    question: str,
    user_id: str = Depends(get_current_user_id)
):
    chain = get_streaming_rag_chain(user_id)

    def generator():
        result = chain.run(question)
        yield result

    return StreamingResponse(
        generator(),
        media_type="text/event-stream"
    )
