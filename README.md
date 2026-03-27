---
title: RAG Document Chatbot
emoji: 📄
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 📄 RAG Document Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot — upload any PDF or text document and ask questions about it. Answers are grounded in your document, not general knowledge.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?logo=langchain&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF7C00?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo

> 🔗 **[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/bytemeuk/rag-doc-chatbot)**
>
> Requires an `OPENAI_API_KEY` Space secret to be set.

## ✨ Features

- **Upload PDF or TXT** documents
- **Grounded answers** — only uses content from your document
- **Conversational memory** — follow-up questions work naturally
- **Free embeddings** — sentence-transformers runs on CPU, no API key needed
- **Source-faithful** — answers "I couldn't find that" if not in the document

## 🧠 How RAG Works

```
Your document
    └─▶ Split into ~800 char chunks (RecursiveCharacterTextSplitter)
            └─▶ Embed each chunk (sentence-transformers/all-MiniLM-L6-v2)
                    └─▶ Store in FAISS vector index

Your question
    └─▶ Embed question (same model)
            └─▶ Find top-4 most similar chunks (cosine similarity)
                    └─▶ Feed chunks + question to GPT-3.5-turbo
                            └─▶ Answer grounded in document context
```

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| LangChain | RAG orchestration (LCEL chain) |
| FAISS | Vector similarity search |
| sentence-transformers | Free CPU embeddings (`all-MiniLM-L6-v2`) |
| OpenAI `gpt-3.5-turbo` | Answer generation |
| Gradio | Web interface |
| Hugging Face Spaces | Deployment |

## 📦 Getting Started

```bash
git clone https://github.com/ByteMe-UK/rag-doc-chatbot.git
cd rag-doc-chatbot

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
python app.py
```

Open `http://localhost:7860`, upload a PDF, and start asking questions.

## 📁 Project Structure

```
rag-doc-chatbot/
├── app.py              ← Gradio UI — upload, process, chat
├── rag/
│   ├── loader.py       ← PDF/TXT loading + text chunking
│   └── chain.py        ← Embeddings, FAISS index, LangChain QA chain
├── .github/
│   └── workflows/
│       └── sync_to_hf.yml  ← Auto-sync to HuggingFace on push
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚢 Deployment (HuggingFace Spaces)

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Gradio**
2. Add `OPENAI_API_KEY` as a Space secret (Settings → Variables and secrets)
3. Add `HFTOKEN` to GitHub repo secrets
4. Push to `main` — GitHub Action syncs to HF automatically

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Part of the [ByteMe-UK](https://github.com/ByteMe-UK) portfolio collection.**
