import re
from sentence_transformers import util

# Keyword-based signals (Legacy/Explicit)
SIGNALS = {
    "stress": ["stress", "overwhelmed", "pressure", "burnout", "tension"],
    "fatigue": ["tired", "exhausted", "drained", "fatigue", "sleepy"],
    "low_mood": ["sad", "down", "depressed", "empty", "hopeless", "grief", "heartbreak", "crying"],
    "anxiety": ["anxious", "worried", "panic", "nervous", "scared", "fear"],
    "sleep_issues": ["sleep", "insomnia", "restless", "wake", "nightmare"],
    "self_worth": ["worthless", "guilt", "shame", "failure", "hate myself"],
    "attention": ["focus", "concentrate", "distracted", "scattered", "brain fog"]
}

# Embedding-based prototypes (Implicit/Semantic)
SIGNAL_PROTOTYPES = {
    "stress": "feeling overwhelmed, pressured, mentally overloaded",
    "low_mood": "sadness, emptiness, hopelessness, emotional heaviness",
    "anxiety": "worry, fear, panic, nervous anticipation",
    "fatigue": "exhaustion, low energy, burnout, tired all the time",
    "sleep_issues": "difficulty sleeping, insomnia, restless nights",
}

PROTOTYPE_EMBEDDINGS = {}

def extract_signals(text, memory, model=None):
    text_lower = text.lower()
    
    # 1. Keyword Extraction (Explicit)
    for signal, keywords in SIGNALS.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text_lower):
                memory["signals"][signal] += 1
                
    # 2. Embedding Extraction (Implicit) - Optional Upgrade
    if model and SIGNAL_PROTOTYPES:
        # Lazy load prototype embeddings
        if not PROTOTYPE_EMBEDDINGS:
            for sig, desc in SIGNAL_PROTOTYPES.items():
                PROTOTYPE_EMBEDDINGS[sig] = model.encode(desc, convert_to_tensor=True)
        
        # Encode user text
        user_emb = model.encode(text, convert_to_tensor=True)
        
        # Compare
        for sig, proto_emb in PROTOTYPE_EMBEDDINGS.items():
            score = util.cos_sim(user_emb, proto_emb).item()
            if score > 0.45:  # Threshold as suggested
                memory["signals"][sig] += 0.5
