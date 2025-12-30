
import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRAG:
    def __init__(self, knowledge_dir="knowledge_base/humanized"):
        print("Loading Knowledge Base Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.knowledge_dir = knowledge_dir
        self.load_knowledge()

    def load_knowledge(self):
        files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        params = []
        for f in files:
            with open(f, 'r') as file:
                text = file.read().strip()
                # Split by double newlines to create chunks
                chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
                self.documents.extend(chunks)
        
        if self.documents:
            print(f"Encoding {len(self.documents)} knowledge chunks...")
            self.embeddings = self.model.encode(self.documents)
        else:
            print("Warning: No knowledge chunks found.")

    def retrieve(self, query, top_k=1, threshold=0.25):
        if not self.documents or self.embeddings is None:
            return None
            
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score < threshold:
            return None
            
        return self.documents[best_idx]


# Singleton instance
knowledge_base = SimpleRAG()

if __name__ == "__main__":
    print("--- Testing Retrieval ---")
    queries = [
        "I feel really panicked and scared",
        "I'm so tired all the time",
        "I can't focus on anything"
    ]
    for q in queries:
        result = knowledge_base.retrieve(q)
        print(f"\nQuery: {q}")
        print(f"Retrieved: {result[:100]}..." if result else "None")
