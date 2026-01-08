import re
from typing import Dict, Any

# Keyword-based signals (matches main.py)
SIGNALS = {
    "stress": ["stress", "overwhelmed", "pressure", "burnout", "tension"],
    "fatigue": ["tired", "exhausted", "drained", "fatigue", "sleepy"],
    "low_mood": ["sad", "down", "depressed", "empty", "hopeless", "grief", "heartbreak", "crying"],
    "anxiety": ["anxious", "worried", "panic", "nervous", "scared", "fear"],
    "sleep_issues": ["sleep", "insomnia", "restless", "wake", "nightmare"],
    "self_worth": ["worthless", "guilt", "shame", "failure", "hate myself"],
    "attention": ["focus", "concentrate", "distracted", "scattered", "brain fog"]
}

def extract_signals(message: str) -> Dict[str, Any]:
    """
    Extracts signals from the user message using keyword matching.
    Returns a dictionary with signal counts.
    """
    signals = {
        "stress": 0,
        "fatigue": 0,
        "low_mood": 0,
        "anxiety": 0,
        "sleep_issues": 0,
        "self_worth": 0,
        "attention": 0
    }
    
    msg_lower = message.lower()
    
    # Keyword Extraction (matches main.py approach)
    for signal, keywords in SIGNALS.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", msg_lower):
                signals[signal] += 1
                
    return signals
