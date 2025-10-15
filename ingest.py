import os
from PyPDF2 import PdfReader

def extract_text_from_file(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Only .pdf and .txt files are supported.")

def load_documents(directory: str = "../documents"):
    docs = []
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filename in os.listdir(directory):
        if filename.endswith((".pdf", ".txt")):
            full_path = os.path.join(directory, filename)
            docs.append(extract_text_from_file(full_path))
    return docs