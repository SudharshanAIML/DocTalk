import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag.retriever import get_retriever
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL")

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
