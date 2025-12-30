import gradio as gr
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# =========================
# CONFIG
# =========================
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# INTENT DESCRIPTIONS
# =========================
INTENTS = {
    "depression": "feeling sad, empty, hopeless, numb, tired, low mood",
    "anxiety": "feeling anxious, worried, nervous, panicky, restless",
    "adhd": "difficulty focusing, distracted, impulsive, restless",
    "autism": "social difficulties, sensory sensitivity, discomfort with change"
}

intent_labels = list(INTENTS.keys())
intent_embeddings = model.encode(list(INTENTS.values()))

# =========================
# SESSION MEMORY
# =========================
def new_memory():
    return {
        "intent_scores": {k: 0.0 for k in INTENTS},
        "turns": [],
        "history": [],
        "assessment_offered": False,
        "active_scale": None
    }

# =========================
# EMBEDDING INTENT UPDATE
# =========================
def update_intent_scores(text, memory):
    user_vec = model.encode([text])
    sims = cosine_similarity(user_vec, intent_embeddings)[0]

    for i, label in enumerate(intent_labels):
        memory["intent_scores"][label] += sims[i]

# =========================
# EMPATHETIC RESPONSE
# =========================
def empathetic_reply(user_text, memory):
    prompt = f"""
You are a warm, friendly mental health companion.

Rules:
- Be empathetic
- Avoid diagnosis
- Avoid clinical language
- Keep it human and gentle
- Ask only ONE open-ended question

Conversation so far:
{memory["turns"]}

User says:
"{user_text}"
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# =========================
# DECIDE WHETHER TO OFFER ASSESSMENT
# =========================
def should_offer_assessment(memory):
    best_intent = max(memory["intent_scores"], key=memory["intent_scores"].get)
    confidence = memory["intent_scores"][best_intent]

    if confidence > 1.5 and not memory["assessment_offered"]:
        return best_intent
    return None

# =========================
# FRIENDLY ASSESSMENT OFFER
# =========================
def offer_assessment(intent):
    return (
        f"It sounds like you might be dealing with some {intent}-related difficulties. "
        "If you’re open to it, I can ask a few short questions that help people reflect on this. "
        "Totally your choice."
    )

# =========================
# SAFETY CHECK PROMPT
# =========================
def safety_check(text):
    danger_words = ["suicide", "kill myself", "end it all", "die"]
    return any(w in text.lower() for w in danger_words)

# =========================
# MAIN CHAT FUNCTION
# =========================
def chat(user_input, state):
    if state is None:
        state = new_memory()

    # Safety check
    if safety_check(user_input):
        state["history"].append(
            {"role": "user", "content": user_input}
        )
        state["history"].append(
            {"role": "assistant", "content":
                "I’m really glad you told me. It sounds like you’re going through a lot. "
                "You deserve support right now. If you’re in immediate danger, please reach out "
                "to local emergency services or a trusted person. You don’t have to handle this alone."
            }
        )
        return state["history"], state

    state["turns"].append(f"User: {user_input}")
    update_intent_scores(user_input, state)

    intent_to_offer = should_offer_assessment(state)
    if intent_to_offer:
        state["assessment_offered"] = True
        response_text = offer_assessment(intent_to_offer)
    else:
        response_text = empathetic_reply(user_input, state)
        state["turns"].append(f"Bot: {response_text}")

    state["history"].append({"role": "user", "content": user_input})
    state["history"].append({"role": "assistant", "content": response_text})

    return state["history"], state

# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="🧠 Hey Attrangi") as demo:
    gr.Markdown("""
    # 🧠 Hey Attrangi  
    *A friendly space to talk and reflect.*

    ⚠️ This is a supportive tool, not a diagnosis.
    """)

    chatbot = gr.Chatbot(height=400)
    state = gr.State()

    user_input = gr.Textbox(placeholder="What’s been on your mind lately?")
    send = gr.Button("Send")

    send.click(chat, [user_input, state], [chatbot, state])
    user_input.submit(chat, [user_input, state], [chatbot, state])

demo.launch()
