"""
RAG chain: embeddings → FAISS vector store → retrieval → LLM answer.

Embeddings: sentence-transformers/all-MiniLM-L6-v2  (free, CPU, ~90MB)
LLM:        Llama 3 via Groq API (free tier)
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}"""),
    ("human", "{question}"),
])


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(docs: list) -> FAISS:
    """Embed document chunks and build a FAISS index."""
    return FAISS.from_documents(docs, _get_embeddings())


def build_chain(vectorstore: FAISS):
    """
    Build the RAG chain using Groq API (free).
    Requires GROQ_API_KEY environment variable.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it as a HuggingFace Space secret."
        )

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=512,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
