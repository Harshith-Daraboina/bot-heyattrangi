import json
from pathlib import Path

MEMORY_FILE = Path("user_memory.json")

def new_memory():
    return {
        "conversation": [],
        "signals": {
            "stress": 0,
            "fatigue": 0,
            "low_mood": 0,
            "anxiety": 0,
            "sleep_issues": 0,
            "self_worth": 0,
            "attention": 0
        },
        "report_offered": False
    }

def load_memory():
    if MEMORY_FILE.exists():
        memory = json.loads(MEMORY_FILE.read_text())
    else:
        memory = new_memory()

    # schema safety
    for k, v in new_memory().items():
        memory.setdefault(k, v)
        
    # Deep merge for signals dict to ensure new signals appear
    if "signals" in memory:
        for k, v in new_memory()["signals"].items():
            memory["signals"].setdefault(k, v)

    return memory

def save_memory(memory):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))

def reset_memory():
    memory = new_memory()
    save_memory(memory)
    return memory

