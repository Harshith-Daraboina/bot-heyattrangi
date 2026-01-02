from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY is not set in environment variables.")
else:
    print(f"✅ GROQ_API_KEY found (starts with: {GROQ_API_KEY[:4]}...)")

BASE_DIR = Path(__file__).parent
MEMORY_FILE = BASE_DIR / "data/user_memory.json"
KB = BASE_DIR / "knowledge_base"

GROQ_MODEL = "llama-3.3-70b-versatile"

SIGNAL_THRESHOLD = 2.0
MAX_FREE_QUESTIONS = 2
