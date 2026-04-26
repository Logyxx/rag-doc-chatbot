"""
RAG chain: embeddings → FAISS vector store → retrieval → LLM answer.

Embeddings: sentence-transformers/all-MiniLM-L6-v2  (free, CPU, ~90MB)
LLM:        Mistral-7B-Instruct via HuggingFace Inference API (free with HF account)

Both use the same HF_TOKEN — no OpenAI key needed.
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPT_TEMPLATE = """<s>[INST] You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question} [/INST]"""


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
    Build the RAG chain using HuggingFace Inference API (free).
    Requires HF_TOKEN environment variable (same token used for deployment).
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it as a HuggingFace Space secret."
        )

    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=hf_token,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt    = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
