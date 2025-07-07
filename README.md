# LLM-Powered Document QA System (RAG)

This project is a lightweight Retrieval-Augmented Generation (RAG) system that enables document-based question answering using a local Large Language Model (LLM). It combines LangChain, Chroma vector database, and Hugging Face models to process PDF documents, retrieve relevant chunks, and generate answers.

---

## Features

- **Offline-capable**: Runs fully locally with Hugging Face models (e.g., LLaMA 3, Falcon).
- **RAG Architecture**: Answers are grounded in your uploaded documents, not hallucinated.
- **PDF Upload Support**: Automatically splits and embeds text for semantic search.
- **Frontend + API**: Streamlit UI + FastAPI backend for a complete user interface.

---

## Project Structure
doc_qa_project/

├── main.py                  # FastAPI backend (upload & QA API)

├── frontend/app.py          # Streamlit frontend UI

├── model_loader.py          # Load Hugging Face model locally

├── rag_chain.py             # RAG logic: split, embed, retrieve, generate

├── requirements.txt         # Python dependencies

├── .gitignore

└── README.md

---

## Installation

```bash
pip install -r requirements.txt
```
Recommended: Use Python 3.9+ with a virtual environment.

## Quickstart

1. Set API key (if using OpenAI embeddings)

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

2. Start FastAPI backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Start Streamlit frontend

```bash
streamlit run frontend/app.py
```

## Supported Models

- meta-llama/Meta-Llama-3-8B-Instruct

- tiiuae/falcon-7b-instruct

- Any AutoModelForCausalLM compatible Hugging Face model

## Example Queries

- “What are the key takeaways from this document?”

- “What conclusions does the author draw in Section 3?”

- “Does this paper mention recent industry trends?”

## TODO

- Support multiple documents

- Add support for Markdown and Word files

- Integrate response time logging

- Add Dockerfile for containerized deployment

## License

MIT License

## Author

Ruoyi Zhu

Feel free to star ⭐ the repo, open issues, or contribute!


