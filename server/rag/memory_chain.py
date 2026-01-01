from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from rag.retriever import get_retriever
import os
from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

def get_conversational_rag_chain(user_id: str):
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0
    )

    retriever = get_retriever(user_id)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
