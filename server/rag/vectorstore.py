import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_user_vectorstore(user_id: str):
    base_path = f"faiss_index/{user_id}"
    embeddings = get_embedding_model()

    if os.path.exists(base_path):
        return FAISS.load_local(base_path, embeddings)
    else:
        return FAISS.from_texts([], embeddings)


def save_user_vectorstore(vectorstore, user_id: str):
    vectorstore.save_local(f"faiss_index/{user_id}")
