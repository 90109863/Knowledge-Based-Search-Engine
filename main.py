# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from ingest import load_documents
from retriever import Retriever
from transformers import pipeline

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load documents and retriever
docs = load_documents()
retriever = Retriever(docs)

# Initialize Flan-T5 for answer synthesis (runs on CPU)
synthesizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # forces CPU
    max_length=150,
    truncation=True
)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = "../documents"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Reload documents and retriever
    global docs, retriever
    docs = load_documents()
    retriever = Retriever(docs)
    return {"message": f"File '{file.filename}' uploaded and indexed!"}

@app.post("/query")
async def query(request: QueryRequest):
    retrieved_docs = retriever.retrieve(request.question, k=3)
    
    if not retrieved_docs:
        return {"answer": "No relevant documents found."}

    context = "\n\n".join(retrieved_docs)

    # Build prompt as per spec: "Using these documents, answer the user’s question succinctly."
    prompt = (
        f"Using these documents, answer the user’s question succinctly.\n\n"
        f"Documents: {context}\n\n"
        f"Question: {request.question}\n"
        f"Answer:"
    )

    try:
        result = synthesizer(prompt)
        answer = result[0]["generated_text"].strip()
        if not answer:
            answer = "I couldn't generate a clear answer from the provided documents."
    except Exception as e:
        answer = "Error during answer generation."

    return {"answer": answer}