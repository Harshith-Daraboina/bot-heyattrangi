import gradio as gr
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

from memory import load_memory, save_memory, reset_memory
from signals import extract_signals
from report import generate_report
from pdf_retriever import pdf_retriever
import pdf_embedder
import os
from pathlib import Path

# Auto-generate embeddings if missing (SKIP on Hugging Face Spaces)
if os.getenv("HF_SPACE") != "true":
    vector_store_path = Path("vector_store/mental_health.index")
    if not vector_store_path.exists():
        print("⚠️ Vector store not found. Generating embeddings...", flush=True)
        pdf_embedder.build_index()
        print("✅ Embeddings generated.", flush=True)
        # Reload retriever
        pdf_retriever.__init__()

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are a warm, emotionally intelligent mental health companion.

Your primary goal is to gently help the user explore their inner experience,
not just to comfort them.


Conversation principles:
- Always acknowledge and validate the user's feelings first
- Then, when emotional safety is established, gently invite deeper sharing
- Ask ONE open-ended question when it would help understanding
- Do not ask questions mechanically or repeatedly
- Avoid yes/no questions
- Avoid clinical or diagnostic language
- If the user expresses trust or vulnerability, deepen the conversation
- Do not get stuck in reassurance loops


Expression Instructions:
- You have a set of facial expressions: EMPATHETIC, NEUTRAL, REFLECTIVE, SAFETY, STRESSED, TIRED.
- Choose the expression based on the USER'S state:
  * TIRED: If user mentions sleep, exhaustion, fatigue, or low energy.
  * STRESSED: If user mentions panic, anxiety, overwhelm, pressure, or fear.
  * SAFETY: Use ONLY if the user mentions self-harm, severe crisis, or specific safety risks.
  * REFLECTIVE: If user is thinking, wondering, asking a question, or exploring a complex idea.
  * EMPATHETIC: If user is sad, grieving, hurt, or needs comfort.
  * NEUTRAL: For greetings, simple acknowledgments, or when no specific strong emotion is present.
- At the very end of your response, strictly output the tag: [EXPRESSION: LISTED_EXPRESSION_NAME].

You are NOT:
- Diagnosing
- Interrogating
- Filling a form

You ARE:
- Helping the user feel understood
- Helping the user reflect
- Helping meaning emerge naturally
"""


CORE_EMOTIONS = ["depressed", "depression", "sad", "hopeless", "stressed", "anxious"]

EXPRESSION_MAP = {
    "EMPATHETIC": "public/bot_expressions/EMPATHETIC.jpg",
    "NEUTRAL": "public/bot_expressions/NEUTRAL.jpg",
    "REFLECTIVE": "public/bot_expressions/REFLECTIVE.jpg",
    "SAFETY": "public/bot_expressions/SAFETY.jpg",
    "STRESSED": "public/bot_expressions/STRESSED.jpg",
    "TIRED": "public/bot_expressions/TIRED.jpg"
}

def requires_exploration(text):
    t = text.lower()
    return any(e in t for e in CORE_EMOTIONS)

def generate_reply(user_text, memory):
    extract_signals(user_text, memory)

    signal_summary = ", ".join(
        f"{k}:{v}" for k, v in memory["signals"].items() if v > 0
    ) or "no strong signals yet"

    recent = memory["conversation"][-6:]

    extra_instruction = ""
    if requires_exploration(user_text):
        extra_instruction = (
            "The user has expressed a strong emotional state. "
            "After validating, gently ask one open-ended question to help them explore it."
        )

    # PDF RAG Retrieval
    print(f"DEBUG: Retrieving for query: {user_text}")
    pdf_context = pdf_retriever.retrieve(user_text)
    print(f"DEBUG: Retrieved {len(pdf_context)} chunks")
    if pdf_context:
        print(f"DEBUG: Top chunk: {pdf_context[0][:100]}...")
    
    knowledge_context = ""
    if pdf_context:
        knowledge_context = (
            f"RELEVANT BACKGROUND KNOWLEDGE:\n"
            f"{' '.join(pdf_context)}\n\n"
            f"INSTRUCTION: Use the above background knowledge to inform your response. "
            f"If the user's input relates to the symptoms, conditions, or situations described in the text, "
            f"incorporate those insights naturally. Do NOT verify the information with the user, just use it to be more specific and helpful."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Observed emotional signals: {signal_summary}"},
        {"role": "system", "content": f"Recent conversation: {recent}"},
        {"role": "system", "content": knowledge_context},
        {"role": "system", "content": extra_instruction},
        {"role": "user", "content": user_text}
    ]


    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.85
    )


    full_response = completion.choices[0].message.content
    
    # Extract expression tag
    expression = "NEUTRAL"
    clean_text = full_response
    
    import re
    match = re.search(r"\[EXPRESSION:\s*([A-Z]+)\]", full_response)
    if match:
        expression = match.group(1).strip()
        # Remove tag from text
        clean_text = full_response.replace(match.group(0), "").strip()
        
    # Validation fallback
    if expression not in EXPRESSION_MAP:
        expression = "NEUTRAL"

    return clean_text, expression


def chat(user_text, history):
    memory = load_memory()

    # LLM needs dict format
    memory["conversation"].append({"role": "user", "content": user_text})

    reply_text, expression_key = generate_reply(user_text, memory)

    # Offer report naturally (once)
    if sum(memory["signals"].values()) >= 6 and not memory["report_offered"]:
        reply_text += (
            "\n\nIf you’d like, I can gently summarize what I’ve noticed so far."
        )
        memory["report_offered"] = True

    memory["conversation"].append({"role": "assistant", "content": reply_text})
    save_memory(memory)

    # UI needs list of dicts format for Gradio >= 5.0
    history = history or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply_text})
    
    avatar_path = EXPRESSION_MAP.get(expression_key, EXPRESSION_MAP["NEUTRAL"])
    
    return history, avatar_path

def show_report():
    memory = load_memory()
    return generate_report(memory)


theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    block_border_width="0px",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)


with gr.Blocks(title="Hey Attrangi", theme=theme) as demo:
    gr.Markdown("## 🧠 Hey Attrangi\nA safe place to talk.")

    
    with gr.Row():
        with gr.Column(scale=1):
            bot_avatar = gr.Image(
                value=EXPRESSION_MAP["NEUTRAL"], 
                interactive=False, 
                show_label=False,
                height=300
            )
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=420)
            user_input = gr.Textbox(placeholder="What's been on your mind?", label="Your Message")
            
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                report_btn = gr.Button("Generate Summary")
    
    report_out = gr.Textbox(lines=12, label="Mental Health Summary")

    send.click(chat, [user_input, chatbot], [chatbot, bot_avatar])
    user_input.submit(chat, [user_input, chatbot], [chatbot, bot_avatar])
    report_btn.click(show_report, None, report_out)
    
    reset_btn = gr.Button("Reset History")
    

    def on_reset():
        reset_memory()
        return [], "", EXPRESSION_MAP["NEUTRAL"]
        
    reset_btn.click(on_reset, None, [chatbot, report_out, bot_avatar])


if __name__ == "__main__":
    print("🚀 Launching Gradio Server...")
    demo.launch()

