
"""Build a Retrieval‑Augmented QA chain."""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

from .model_loader import load_llm

DB_DIR = Path(__file__).parent / "chroma_db"

def ingest_document(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(path).load()
    else:
        docs = TextLoader(path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=str(DB_DIR)
    )
    vectordb.persist()
    return len(chunks)

def build_qa_chain():
    vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=OpenAIEmbeddings())
    llm = load_llm()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain
