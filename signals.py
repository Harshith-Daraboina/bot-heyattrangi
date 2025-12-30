import re

SIGNALS = {
    "stress": ["stress", "overwhelmed", "pressure", "burnout", "tension"],
    "fatigue": ["tired", "exhausted", "drained", "fatigue", "sleepy"],
    "low_mood": ["sad", "down", "depressed", "empty", "hopeless", "grief", "heartbreak", "crying"],
    "anxiety": ["anxious", "worried", "panic", "nervous", "scared", "fear"],
    "sleep_issues": ["sleep", "insomnia", "restless", "wake", "nightmare"],
    "self_worth": ["worthless", "guilt", "shame", "failure", "hate myself"],
    "attention": ["focus", "concentrate", "distracted", "scattered", "brain fog"]
}

def extract_signals(text, memory):
    text = text.lower()
    for signal, keywords in SIGNALS.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text):
                memory["signals"][signal] += 1
