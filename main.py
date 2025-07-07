
"""FastAPI backend for file upload and Q&A."""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil, tempfile

from rag_chain import ingest_document, build_qa_chain

app = FastAPI(title="Document QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = None  # lazy init

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    num_chunks = ingest_document(tmp_path)
    return {"status": "ok", "chunks": num_chunks}

@app.post("/ask")
async def ask(question: str = Form(...)):
    global qa_chain
    if qa_chain is None:
        qa_chain = build_qa_chain()
    res = qa_chain({"query": question})
    return {"answer": res["result"]}
