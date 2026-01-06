from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

PDF_DIR = Path("knowledge_base")
INDEX_DIR = Path("vector_store")
INDEX_DIR.mkdir(exist_ok=True)

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_path):
    print(f"Extracting text from: {file_path}")
    text = ""
    try:
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def build_index():
    all_chunks = []
    metadata = []

    files = sorted(list(PDF_DIR.rglob("*.pdf")) + list(PDF_DIR.rglob("*.txt")))
    if not files:
        print(f"No PDF or TXT files found in {PDF_DIR}")
        return

    for file_path in files:
        text = extract_text(file_path)
        if not text:
            print(f"No text extracted from {file_path.name}")
            continue
            
        chunks = chunk_text(text)
        print(f"  > {file_path.name}: {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "source": file_path.name
            })

    if not all_chunks:
        print("No content to index.")
        return

    print(f"Encoding {len(all_chunks)} total chunks...")
    embeddings = MODEL.encode(all_chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "mental_health.index"))
    
    # Save texts and metadata
    json.dump(
        {"texts": all_chunks, "meta": metadata},
        open(INDEX_DIR / "mental_health.json", "w"),
        indent=2
    )

    print(f"✅ Embedded {len(all_chunks)} chunks from {len(files)} files")

if __name__ == "__main__":
    build_index()
