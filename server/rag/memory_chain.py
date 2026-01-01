from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rag.retriever import get_retriever
import os
from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("GEMINI_MODEL")

def get_conversational_rag_chain(user_id: str):
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever = get_retriever(user_id)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
