"""
Document loading and chunking for the RAG pipeline.

Supports:
  - PDF files  (.pdf)  via pypdf
  - Plain text (.txt)  read directly
"""

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_and_split(file_path: str) -> list:
    """
    Load a PDF or TXT file and split into overlapping chunks.

    Args:
        file_path: absolute path to the uploaded file

    Returns:
        List of LangChain Document objects (each has .page_content and .metadata)
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Upload a .pdf or .txt file.")

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
