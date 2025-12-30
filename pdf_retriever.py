import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

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
            self.store = json.load(open(STORE_PATH, "r"))
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            print("Warning: PDF Vector Store not found. Run pdf_embedder.py first.")

    def retrieve(self, query, top_k=3):
        if not self.index or not self.model:
            return []

        q_emb = self.model.encode([query])
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.store["texts"]):
                # Optional: Filter by distance if needed, but for now just return top_k
                results.append(self.store["texts"][idx])

        return results

# Singleton instance
pdf_retriever = PDFRetriever()

if __name__ == "__main__":
    print("--- Testing PDF Retrieval ---")
    queries = [
        "What are the symptoms of GAD?",
        "I feel panicked and scared",
        "How is autism diagnosed?",
        "I can't focus on anything"
    ]
    for q in queries:
        result = pdf_retriever.retrieve(q, top_k=1)
        print(f"\nQuery: {q}")
        print(f"Retrieved: {result[0][:150]}..." if result else "None")
