
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("📚 Document QA (RAG)")

uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "md"])
if uploaded and st.button("Ingest document"):
    files = {"file": (uploaded.name, uploaded.getvalue())}
    res = requests.post(f"{API_URL}/upload", files=files)
    st.success(f"Ingested {res.json().get('chunks', 0)} chunks.")

question = st.text_input("Ask a question about the document")
if question and st.button("Ask"):
    res = requests.post(f"{API_URL}/ask", data={"question": question})
    st.write(res.json().get("answer", "No answer"))
