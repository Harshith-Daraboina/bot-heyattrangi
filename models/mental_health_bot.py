import json
from pathlib import Path
from groq import Groq

# =========================
# CONFIG
# =========================
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

BASE = Path("knowledge_base")

SCALE_MAP = {
    "adhd": BASE / "adhd/vanderbilt_structured.json",
    "anxiety": BASE / "anxiety/gad7_structured.json",
    "depression": BASE / "depression/phq9_structured.json",
    "autism": BASE / "autism/aq_structured.json"
}

# =========================
# 1. INTENT DETECTION
# =========================
def detect_intent(user_text: str):
    text = user_text.lower()

    depression_words = [
        "sad", "unhappy", "depressed", "hopeless", "empty",
        "numb", "tired of life", "worthless", "low", "down"
    ]

    anxiety_words = [
        "anxious", "anxiety", "worried", "panic", "nervous",
        "overthinking", "scared", "fear", "tense", "restless"
    ]

    adhd_words = [
        "focus", "attention", "concentrate", "distracted",
        "hyper", "impulsive", "restless", "forgetful"
    ]

    autism_words = [
        "autism", "social", "overstimulated", "sensory",
        "routine", "eye contact", "social cues", "awkward"
    ]

    if any(word in text for word in depression_words):
        return "depression"
    if any(word in text for word in anxiety_words):
        return "anxiety"
    if any(word in text for word in adhd_words):
        return "adhd"
    if any(word in text for word in autism_words):
        return "autism"

    return None

# =========================
# 2. LOAD SCALE
# =========================
def load_scale(scale_key):
    with open(SCALE_MAP[scale_key], "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# 3. QUESTIONNAIRE ENGINE
# =========================
def run_questionnaire(scale):
    print(f"\n🧠 {scale['scale']}")
    print(f"Timeframe: {scale.get('timeframe','')}\n")

    responses = {}
    for q in scale["questions"]:
        print(f"{q['id']}. {q['text']}")
        val = int(input("Enter response (0–3 or 1–4): "))
        responses[q["id"]] = val
    return responses

# =========================
# 4. SCORING
# =========================
def score_scale(scale, responses):
    if scale["scoring"]["method"] == "sum":
        total = sum(responses.values())
        return total
    elif scale["scoring"]["method"] == "symptom_count":
        threshold = scale["scoring"]["threshold"]
        # Count how many responses match or exceed the threshold
        count = sum(1 for val in responses.values() if val >= threshold)
        return count
    raise ValueError("Unknown scoring method")

# =========================
# 5. RISK CLASSIFICATION
# =========================
def classify_risk(scale, score):
    for level, (low, high) in scale["severity_cutoffs"].items():
        if low <= score <= high:
            return level
    return "unknown"

# =========================
# 6. SAFETY CHECK
# =========================
def safety_check(scale_key, responses):
    if scale_key == "depression":
        if responses.get(9, 0) > 0:
            return True
    return False

# =========================
# 7. SUGGESTION ENGINE
# =========================
def load_suggestions():
    with open(BASE / "explanations/coping_suggestions.txt", "r") as f:
        return f.read()

# =========================
# 8. LLM EXPLANATION
# =========================
def llm_explain(scale, score, risk, suggestions):
    prompt = f"""
You are a mental health support assistant.

Scale: {scale['scale']}
Score: {score}
Risk Level: {risk}

Explain the results in a calm, empathetic, non-diagnostic way.
Reference DSM concepts gently.
Offer general coping strategies.

Coping suggestions:
{suggestions}

IMPORTANT:
- Do NOT diagnose
- Be supportive and respectful
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

# =========================
# 9. MAIN APP
# =========================
def run_bot():
    print("🧠 Mental Health Screening Assistant")
    user_text = input("Tell me what you're struggling with:\n> ")

    intent = detect_intent(user_text)
    if not intent:
        print("I couldn't identify a specific area. Please try again.")
        return

    scale = load_scale(intent)
    responses = run_questionnaire(scale)

    if safety_check(intent, responses):
        print("\n⚠️ IMPORTANT")
        print("It sounds like you may be experiencing significant distress.")
        print("Please consider reaching out to a mental health professional or emergency services.")
        return

    score = score_scale(scale, responses)
    risk = classify_risk(scale, score)

    print(f"\n📊 Score: {score}")
    print(f"⚖️ Risk Level: {risk}")

    suggestions = load_suggestions()
    explanation = llm_explain(scale, score, risk, suggestions)

    print("\n🧠 Explanation\n")
    print(explanation)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_bot()
