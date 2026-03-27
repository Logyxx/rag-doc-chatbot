"""
RAG chain: embeddings → FAISS vector store → retrieval → LLM answer.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (free, CPU, ~90MB)
LLM:        OpenAI gpt-3.5-turbo (requires OPENAI_API_KEY env var)
"""

import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load sentence-transformers embeddings (cached after first download)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(docs: list) -> FAISS:
    """Embed document chunks and build a FAISS index."""
    embeddings = _get_embeddings()
    return FAISS.from_documents(docs, embeddings)


def build_chain(vectorstore: FAISS):
    """
    Build a retrieval-augmented generation chain.

    Returns a LangChain Runnable that accepts {"question": str}
    and returns a string answer.

    Raises EnvironmentError if OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it as an environment variable or a HuggingFace Space secret."
        )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
