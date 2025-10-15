# backend/retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

class Retriever:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not docs:
            # No documents? Create a dummy index to avoid crash
            self.index = None
            self.embeddings = None
            return

        embeddings = self.model.encode(docs, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        self.embeddings = embeddings
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if self.index is None or len(self.docs) == 0:
            return []
        
        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        _, I = self.index.search(query_vec, k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results