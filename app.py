"""
RAG Document Chatbot — Gradio interface.

Upload a PDF or TXT file, then ask questions about it.
Answers are grounded in the document — no hallucination.

Requires: HF_TOKEN environment variable (HuggingFace token — free)
"""

import os
import gradio as gr

from rag.loader import load_and_split
from rag.chain import build_vectorstore, build_chain

# Module-level state — rebuilt each time a new document is uploaded
_chain = None


def process_document(file) -> str:
    """Load, chunk, embed, and index the uploaded file."""
    global _chain

    if file is None:
        return "No file uploaded."

    try:
        docs = load_and_split(file.name)
        if not docs:
            return "Could not extract text from the document."

        vectorstore = build_vectorstore(docs)
        _chain = build_chain(vectorstore)

        chunk_count = len(docs)
        return f"✅ Document loaded — {chunk_count} chunks indexed. You can now ask questions below."

    except EnvironmentError as e:
        return f"⚠️ {e}"
    except Exception as e:
        return f"❌ Error processing document: {e}"


def answer_question(question: str, history: list) -> tuple[str, list]:
    """Run a question through the RAG chain and return the answer."""
    if not question.strip():
        return "", history

    if _chain is None:
        history.append((question, "Please upload a document first."))
        return "", history

    try:
        answer = _chain.invoke(question)
    except Exception as e:
        answer = f"❌ Error: {e}"

    history.append((question, answer))
    return "", history


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="RAG Document Chatbot") as demo:
    gr.Markdown(
        """
        # 📄 RAG Document Chatbot
        Upload a **PDF or TXT** document, then ask questions about it.
        Answers are grounded in your document — not general knowledge.

        > Powered by **Mistral-7B** via HuggingFace Inference API — completely free.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Document (PDF or TXT)",
                file_types=[".pdf", ".txt"],
            )
            upload_btn = gr.Button("📥 Load Document", variant="primary")
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload a document to get started...",
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=420, type="tuples")
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    label="Question",
                    scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

    gr.Examples(
        examples=[
            ["What is the main topic of this document?"],
            ["Summarise the key points."],
            ["What conclusions does the author draw?"],
        ],
        inputs=question_box,
        label="Example questions",
    )

    upload_btn.click(fn=process_document, inputs=file_input, outputs=status_box)
    ask_btn.click(fn=answer_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    question_box.submit(fn=answer_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])

    gr.Markdown(
        "---\nBuilt by [Laksh Menroy](https://github.com/lakshmenroy) · "
        "[Logyxx](https://github.com/Logyxx) portfolio"
    )

demo.launch()

