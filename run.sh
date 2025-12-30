#!/bin/bash

# 1. Install dependencies (if not already installed)
# pip install -r requirements.txt
# Or manually:
# pip install groq gradio sentence-transformers faiss-cpu pypdf scikit-learn

# 2. Generate Embeddings (if index doesn't exist)
if [ ! -f "vector_store/mental_health.index" ]; then
    echo "🧠 Generating Knowledge Base Embeddings..."
    python3 pdf_embedder.py
else
    echo "✅ Knowledge Base found. Skipping embedding generation."
    echo "(Run 'rm vector_store/mental_health.index' to regenerate if you added new PDFs)"
fi

# 3. specific for the linux 
export GRADIO_SERVER_NAME="0.0.0.0"

# 4. Run the App
echo "🚀 Starting Hey Attrangi Bot..."
python3 app.py
