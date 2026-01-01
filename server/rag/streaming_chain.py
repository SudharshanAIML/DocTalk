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
MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming tokens to a queue."""
    
    def __init__(self, queue: Queue):
        self.queue = queue
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)
    
    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)  # Signal end of stream


def get_streaming_rag_chain(user_id: str):
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0,
        streaming=True
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer using ONLY the context.

Context:
{context}

Question:
{question}
"""
    )

    retriever = get_retriever(user_id)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )


def stream_rag_response(user_id: str, question: str):
    """Generator that yields tokens as they are generated."""
    queue = Queue()
    callback = StreamingCallbackHandler(queue)
    
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer using ONLY the context.

Context:
{context}

Question:
{question}
"""
    )

    retriever = get_retriever(user_id)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
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
