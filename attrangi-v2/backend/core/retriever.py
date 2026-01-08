import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

# Adjust path assuming this runs from backend/
INDEX_PATH = Path("vector_store/mental_health.index")
STORE_PATH = Path("vector_store/mental_health.json")

class PDFRetriever:
    def __init__(self):
        self.index = None
        self.store = None
        self.model = None
        
        if INDEX_PATH.exists() and STORE_PATH.exists():
            print("Loading PDF Vector Store...")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(STORE_PATH, "r") as f:
                self.store = json.load(f)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            print(f"Warning: PDF Vector Store not found at {INDEX_PATH.absolute()}.")

    def retrieve(self, query: str, top_k: int = 3):
        if not self.index or not self.model:
            return []

        q_emb = np.array(self.model.encode([query]), dtype="float32")
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.store["texts"]):
                # Optional: Filter by distance if needed, but for now just return top_k
                results.append(self.store["texts"][idx])

        return results

# Singleton instance
retriever = PDFRetriever()
