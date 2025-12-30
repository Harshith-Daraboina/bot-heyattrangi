from pathlib import Path

GROQ_API_KEY = "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"

BASE_DIR = Path(__file__).parent
MEMORY_FILE = BASE_DIR / "data/user_memory.json"
KB = BASE_DIR / "knowledge_base"

GROQ_MODEL = "llama-3.3-70b-versatile"

SIGNAL_THRESHOLD = 2.0
MAX_FREE_QUESTIONS = 2
