from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag.retriever import get_retriever

def get_rag_chain(user_id: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant.
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"Answer not found in uploaded documents."
"""
    )

    retriever = get_retriever(user_id)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain
