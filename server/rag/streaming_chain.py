import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from rag.retriever import get_retriever
from dotenv import load_dotenv
from queue import Queue
from threading import Thread

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

STREAM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer using ONLY the context. The context may come from MULTIPLE documents — cover ALL of them.

FORMAT your response using markdown:
- Use ## headers for sections
- Use **bold** for key terms
- Use bullet points for lists
- Use markdown tables for numerical/metric data

Context:
{context}

Question:
{question}

Answer (well-formatted markdown):
"""
)


def _extract_text(token) -> str:
    """
    Normalize a streamed token to plain text.

    Newer Gemini models return content as a list of typed blocks
    (e.g. [{"type": "text", "text": "..."}]) rather than a plain string,
    interleaved with non-text blocks (like thought-signature metadata) that
    carry no visible text — those must be skipped, not stringified.
    """
    if isinstance(token, str):
        return token
    if isinstance(token, list):
        parts = []
        for item in token:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type", "text") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming tokens to a queue."""

    def __init__(self, queue: Queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        text = _extract_text(token)
        if text:
            self.queue.put(text)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)  # Signal end of stream


def stream_rag_response(user_id: str, question: str):
    """Generator that yields tokens as they are generated, using Gemini."""
    queue = Queue()
    callback = StreamingCallbackHandler(queue)

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        streaming=True,
        google_api_key=GOOGLE_API_KEY,
        callbacks=[callback]
    )

    retriever = get_retriever(user_id)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": STREAM_PROMPT}
    )

    def run_chain():
        try:
            chain.run(question)
        except Exception as e:
            queue.put(f"\n[ERROR]: {str(e)}")
            queue.put(None)

    # Run chain in background thread
    thread = Thread(target=run_chain)
    thread.start()

    # Yield tokens as they come
    while True:
        token = queue.get()
        if token is None:
            break
        yield token

    thread.join()
