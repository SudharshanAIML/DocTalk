from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from rag.retriever import get_retriever
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Custom prompt to rephrase questions for better document retrieval
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question 
to be a standalone question that includes specific keywords for document search.

If the user asks about "myself", "me", "my background", "my resume", "my profile", etc., 
rephrase it to search for personal information, resume, skills, education, experience, projects, 
name, profile, background in the uploaded documents.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question with search keywords:""")

# Custom QA prompt to answer from documents (without web context)
QA_PROMPT_DOCS_ONLY = PromptTemplate.from_template("""
You are an AI assistant that answers questions based ONLY on the provided context from uploaded documents.

IMPORTANT RULES:
1. Answer ONLY using information from the context below.
2. If the user asks about "myself" or "me", look for resume, profile, or personal information in the context.
3. If the information is not in the context, say "I couldn't find this information in your uploaded documents."
4. Search across ALL provided context chunks — they may come from DIFFERENT documents. Cover information from EVERY document, not just one.
5. Be specific and cite which document the information comes from when possible.

FORMATTING RULES (very important):
- Use **markdown** formatting in your response.
- Use `##` headers to organize different topics or models.
- Use **bold** for key terms, model names, and important values.
- Use bullet points (`-`) for listing features, characteristics, or details.
- When presenting numerical data or performance metrics, use a **markdown table** with clear column headers.
- Keep the response well-structured and easy to scan.

Context from uploaded documents:
{context}

Question: {question}

Answer (comprehensive, well-formatted markdown covering ALL documents):""")

# Custom QA prompt with web context
QA_PROMPT_WITH_WEB = """
You are an AI assistant that answers questions using BOTH uploaded documents AND web search results.

IMPORTANT RULES:
1. Use information from BOTH the uploaded documents AND web search results.
2. Prioritize uploaded documents for personal/specific information.
3. Use web results to supplement with current/general information.
4. ALWAYS cite your sources - indicate whether information comes from "[Document: filename]" or "[Web: url]".
5. If information conflicts between sources, mention both perspectives.

FORMATTING RULES:
- Use **markdown** formatting in your response.
- Use `##` headers to organize different topics.
- Use **bold** for key terms and important values.
- Use bullet points for lists.
- When presenting data, use markdown tables.
- Clearly distinguish between document-based and web-based information.

=== UPLOADED DOCUMENTS ===
{context}

=== WEB SEARCH RESULTS ===
{web_context}

Question: {question}

Answer (comprehensive, citing sources from both documents and web):"""


def get_conversational_rag_chain(user_id: str, web_context: str = "", filter_document_ids=None):
    """
    Create a conversational RAG chain that searches across user documents.
    Uses custom prompts to better handle personal queries like "tell me about myself".
    
    Args:
        user_id: User identifier
        web_context: Optional web search context to include
        filter_document_ids: Optional list of document IDs to restrict search to
    """
    llm = ChatGroq(
        model=MODEL,
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    # Get retriever; pass filter_document_ids to restrict to selected docs
    retriever = get_retriever(user_id, k=8, filter_document_ids=filter_document_ids)
    
    # Choose prompt based on whether web context is provided
    if web_context:
        # Create a dynamic prompt with web context
        qa_prompt = PromptTemplate.from_template(
            QA_PROMPT_WITH_WEB.replace("{web_context}", web_context)
        )
        logger.info(f"Creating conversational chain for user: {user_id} WITH web context")
    else:
        qa_prompt = QA_PROMPT_DOCS_ONLY
        logger.info(f"Creating conversational chain for user: {user_id} (documents only)")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=True
    )
