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
    "attention": ["focus", "concentrate", "distracted", "scattered", "brain fog"],
    # Note: Violence and Vulnerability are handled specifically via PROTOTYPES or Regex, 
    # but keeping keys here for structure is fine.
    "violence_intent": [], 
    "vulnerability": ["confused", "don't know", "unsure", "maybe", "scared", "honest"],
}

# Embedding-based prototypes (Implicit/Semantic)
SIGNAL_PROTOTYPES = {
    "stress": "feeling overwhelmed, pressured, mentally overloaded",
    "low_mood": "sadness, emptiness, hopelessness, emotional heaviness",
    "anxiety": "worry, fear, panic, nervous anticipation",
    "fatigue": "exhaustion, low energy, burnout, tired all the time",
    "sleep_issues": "difficulty sleeping, insomnia, restless nights",
    "violence_intent": "intent to physically harm or kill another person, violent rage, making threats",
    "vulnerability": "admitting uncertainty or confusion while emotionally open, sharing something personal without strong distress",
}

RESPONSE_MODE_PROTOTYPES = {
    "answer": (
        "asking for a clear reason or explanation, frustrated by questions, "
        "wants a direct answer, says just answer or why does this happen"
    ),
    "explore": (
        "open to discussing feelings, reflecting, understanding more deeply, "
        "curious about patterns"
    ),
    "vent": (
        "expressing hurt, anger, frustration, wants to be heard, not asking for solutions"
    ),
}

# Advanced Logic Constants
VIOLENCE_PATTERNS = [
    r"\bi will (kill|hurt|attack|murder|smash)\b",
    r"\bi want to (kill|hurt|attack|murder|smash)\b",
    r"\bi am going to (kill|hurt|attack|murder|smash)\b",
    r"\bgonna (kill|hurt|attack|murder|smash)\b",
]

THRESHOLDS = {
    "violence_intent": 0.70,
    "low_mood": 0.50,
    "anxiety": 0.50,
    "vulnerability": 0.55,
}

NEGATIONS = ["not", "don't", "never", "wouldn't", "won't", "cant", "can't"]

PROTOTYPE_EMBEDDINGS = {}
RESPONSE_MODE_EMBEDDINGS = {}

def decay_signals(memory, decay=0.85):
    """Reduces signal intensity to represent emotional momentum."""
    for k in memory["signals"]:
        memory["signals"][k] *= decay

def is_negated(text, keyword, window=3):
    """Checks if a keyword is preceded by a negation in a small window."""
    tokens = text.lower().split()
    # Simple token check - strict matching
    # Find all indices of keyword (substring match in token)
    matches = [i for i, t in enumerate(tokens) if keyword in t]
    
    for i in matches:
        start = max(0, i - window)
        # check negation in the window before the keyword
        if any(n in tokens[start:i] for n in NEGATIONS):
            return True
    return False

def extract_signals(text, memory, model=None):
    text_lower = text.lower()
    
    # 0. Apply Decay
    decay_signals(memory)
    
    # 1. Hard Violence Override (Regex)
    # Check regex patterns first for maximum safety
    for pattern in VIOLENCE_PATTERNS:
        if re.search(pattern, text_lower):
            # Check negation implicitly? Regex usually captures positive intent, 
            # but let's be careful. If regex matches "will not kill", it won't match "will kill" usually.
            # But "i will not kill" contains "will ... kill" if regex is loose.
            # The patterns defined are specific: \bi will kill\b.
            # "I will not kill" -> "will not kill". It WON'T match "will kill" unless the regex allows gaps.
            # Our regex is strict: \bi will (kill...)\b. So "will not kill" is safe.
            
            memory["signals"]["violence_intent"] = 1.0
            memory["stage"] = "safety"
            memory["lock_stage"] = True
            return # Exit immediately to preventing softening
            
    # 2. Keyword Extraction (Explicit) with Negation
    for signal, keywords in SIGNALS.items():
        if signal == "violence_intent": continue # Skip keyword list for violence, rely on regex/embedding
        
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text_lower):
                if not is_negated(text_lower, kw):
                    memory["signals"][signal] += 1
                
    # 3. Embedding Extraction (Implicit)
    if model and SIGNAL_PROTOTYPES:
        # Lazy load prototype embeddings
        if not PROTOTYPE_EMBEDDINGS:
            for sig, desc in SIGNAL_PROTOTYPES.items():
                PROTOTYPE_EMBEDDINGS[sig] = model.encode(desc, convert_to_tensor=True)
        
        # Encode user text
        user_emb = model.encode(text, convert_to_tensor=True)
        
        # Compare
        for sig, proto_emb in PROTOTYPE_EMBEDDINGS.items():
            threshold = THRESHOLDS.get(sig, 0.45) # Default 0.45
            score = util.cos_sim(user_emb, proto_emb).item()
            
            if score > threshold:
                # Double check negotiation for embedding? Hard to do. 
                # Trust the threshold and semantic meaning.
                
                # Special Safety Check for Violence Embedding
                if sig == "violence_intent":
                     # Extra high threshold was processed.
                     memory["signals"]["violence_intent"] = 1.0
                     memory["stage"] = "safety"
                     memory["lock_stage"] = True
                     return
                
                # Vulnerability Check - Don't trigger if high distress
                if sig == "vulnerability":
                     if memory["signals"].get("violence_intent", 0) > 0:
                         continue
                
                memory["signals"][sig] += 0.5

def detect_response_mode(text, model, min_confidence=0.55):
    if not RESPONSE_MODE_EMBEDDINGS:
            for mode, desc in RESPONSE_MODE_PROTOTYPES.items():
                RESPONSE_MODE_EMBEDDINGS[mode] = model.encode(desc, convert_to_tensor=True)
    
    user_emb = model.encode(text, convert_to_tensor=True)
    scores = {}
    for mode, proto_emb in RESPONSE_MODE_EMBEDDINGS.items():
        scores[mode] = util.cos_sim(user_emb, proto_emb).item()
    
    best_mode = max(scores, key=scores.get)
    if scores[best_mode] < min_confidence:
        return "explore"
        
    return best_mode
