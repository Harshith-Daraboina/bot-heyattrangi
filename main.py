from nicegui import ui, app, run
from pathlib import Path
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from memory import load_memory, save_memory, reset_memory
from signals import extract_signals
from report import generate_report
import pdf_retriever as pdf_retriever_module
import pdf_embedder
import importlib
import asyncio
import re

# ==============================
# CONFIG & INIT
# ==============================
VECTOR_PATH = Path("vector_store/mental_health.index")

# Initialize Retriever (Lazy Load)
def get_retriever():
    if not VECTOR_PATH.exists():
        pdf_embedder.build_index()
    importlib.reload(pdf_retriever_module)
    return pdf_retriever_module.pdf_retriever

retriever = get_retriever()
client = Groq(api_key=GROQ_API_KEY)

EXPRESSION_MAP = {
    "EMPATHETIC": "public/bot_expressions/EMPATHETIC.jpg",
    "STRESSED": "public/bot_expressions/STRESSED.jpg",
    "TIRED": "public/bot_expressions/TIRED.jpg",
    "REFLECTIVE": "public/bot_expressions/REFLECTIVE.jpg",
    "NEUTRAL": "public/bot_expressions/NEUTRAL.jpg",
    "SAFETY": "public/bot_expressions/SAFETY.jpg",
}

SYSTEM_PROMPT = """
You are a warm, emotionally intelligent mental health companion.

Your role is to help the user feel heard and gently understand what they are going through.
You are having a real human conversation, not conducting therapy or an interview.

────────────────────────
Core Style Rules
────────────────────────
- Respond like a real person, not a therapist or chatbot
- Use natural, everyday language
- Do NOT use generic empathy phrases such as:
  “That can be really tough”
  “I’m sorry you’re going through this”
  “It sounds like…”
- Vary sentence structure and emotional framing each turn
- Never interrogate, rush, or overwhelm the user
- Do NOT reintroduce yourself once the conversation has started
- If the user says “hi” again mid-conversation, treat it as continuation, not a restart

────────────────────────
How to Respond
────────────────────────
1. Acknowledge the user’s situation or feeling in a specific, human way  
   (reflect *their* words, not a template)

2. Choose ONE of the following paths:
   - **Soft invitation (no question)** when emotions are heavy  
     Example: “We can sit with this for a moment if you want.”
   - **One targeted, open-ended question** when exploration helps
   - **Reflection only** when the user is already sharing deeply

3. If the user’s input is short or vague, ask ONE clarifying question  
   (avoid “Tell me more”)

────────────────────────
Question Guidance
────────────────────────
When asking a question, explore only ONE dimension:
- Cause → “What do you think set this off?”
- Impact → “What’s been hardest day to day?”
- Meaning → “What has this made you question?”
- Attachment → “What do you miss, or don’t miss?”

Avoid abstract or multiple-choice questions.
Never ask more than ONE question in a turn.

────────────────────────
Conversation Flow Control
────────────────────────
- If the conversation slows, gently suggest a relevant direction based on prior context
- Do not repeat the same type of question in consecutive turns
- Allow silence and space when appropriate

────────────────────────
Facial Expression Selection
────────────────────────
Choose ONE expression based on the user’s current state:
- EMPATHETIC → sadness, grief, emotional pain
- STRESSED → anxiety, overwhelm, pressure
- TIRED → exhaustion, low energy, sleep difficulty
- REFLECTIVE → thinking aloud, meaning-making
- SAFETY → explicit self-harm or severe crisis
- NEUTRAL → greetings or low emotional intensity

────────────────────────
Output Rule
────────────────────────
At the very end of every response, output exactly:
[EXPRESSION: ONE_EXPRESSION]

"""


# ==============================
# LOGIC
# ==============================
def update_stage(memory):
    score = sum(memory["signals"].values())
    if score >= 5:
        memory["stage"] = "synthesis"
    elif score >= 2:
        memory["stage"] = "exploration"
    else:
        memory["stage"] = "opening"

def generate_reply_sync(user_text, memory):
    # No delay as requested
    
    extract_signals(user_text, memory, model=pdf_embedder.MODEL)
    update_stage(memory)

    recent = memory["conversation"][-6:]
    pdf_context = retriever.retrieve(user_text) if retriever else []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Conversation stage: {memory['stage']}"},
        {"role": "system", "content": f"Recent conversation: {recent}"},
    ]

    if pdf_context:
        messages.append({
            "role": "system",
            "content": (
                "Background mental health knowledge (use gently, do not quote):\n"
                + "\n".join(pdf_context)
            )
        })

    messages.append({
        "role": "system",
        "content": "If the user names a new emotion, respond in a new way."
    })

    messages.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.85
    )

    full = completion.choices[0].message.content
    
    # Robust Regex: Matches [EXPRESSION: TAG] OR [TAG]
    match = re.search(r"\[(?:EXPRESSION:\s*)?([A-Z]+)\]", full)
    expression = match.group(1) if match else "NEUTRAL"
    
    # Remove the tag from the text shown to user
    clean = re.sub(r"\[(?:EXPRESSION:\s*)?([A-Z]+)\]", "", full).strip()

    return clean, expression

# ==============================
# UI
# ==============================
@ui.page('/')
async def main_page():
    # Session State
    if 'memory' not in app.storage.user:
        app.storage.user['memory'] = load_memory()
    if 'expression' not in app.storage.user:
        app.storage.user['expression'] = "NEUTRAL"

    # Fix Layout: Left Drawer spans full height (lHh LpR lFf)
    ui.query('.q-layout').props('view="lHh LpR lFf"')

    # Theme Styling
    ui.add_head_html("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');
            body { font-family: 'Outfit', sans-serif; background-color: #0F172A; color: #F1F5F9; }
            .q-drawer { background-color: #020617 !important; border-right: 1px solid #1E293B; }
            .chat-user { background-color: #1E3A8A; color: white; border-radius: 16px 16px 4px 16px; padding: 16px 20px; font-size: 1.125rem; border: 1px solid #2563EB; max-width: 80%; margin-left: auto; }
            .chat-bot { background-color: #1E293B; color: #E2E8F0; border-radius: 16px 16px 16px 4px; padding: 16px 20px; font-size: 1.125rem; border: 1px solid #334155; max-width: 80%; }
            .input-area { background-color: #0F172A; border-top: 1px solid #1E293B; }
        </style>
    """)

    # --- SIDEBAR ---
    with ui.left_drawer(value=True).props('width=450').classes('column justify-between p-6'):
        with ui.column().classes('w-full items-center gap-4'):
            
            
            
            avatar = ui.image(EXPRESSION_MAP["NEUTRAL"]).classes('rounded-2xl w-full shadow-lg border-4 border-slate-800').props('fit=cover')
            avatar.bind_source_from(app.storage.user, 'expression', backward=lambda x: EXPRESSION_MAP.get(x, EXPRESSION_MAP["NEUTRAL"]))

            with ui.row().classes('w-full justify-center mt-2'):
                ui.badge().bind_text_from(app.storage.user, 'expression', backward=lambda x: f"{x} MODE").classes('bg-blue-900 text-blue-100 px-3 py-1 text-sm')

        with ui.column().classes('w-full gap-2'):
            async def generate_summary():
                ui.notify('Generating summary...', color='info')
                report = await run.io_bound(generate_report, app.storage.user['memory'])
                with ui.dialog() as dialog, ui.card().classes('bg-slate-800 text-slate-100 w-full max-w-2xl'):
                    ui.label('Mental Health Summary').classes('text-xl font-bold mb-4')
                    ui.textarea(value=report).props('readonly autogrow').classes('w-full bg-slate-900 border-slate-700 rounded-lg p-2')
                    ui.button('Close', on_click=dialog.close).props('no-caps unelevated').classes('bg-slate-700 hover:bg-slate-600 text-white rounded-lg mt-4')
                dialog.open()

            async def reset_chat():
                reset_memory()
                app.storage.user['memory'] = load_memory()
                app.storage.user['expression'] = "NEUTRAL"
                ui.navigate.reload()

            ui.button('Generate Summary', on_click=generate_summary).props('no-caps unelevated').classes('w-full h-14 text-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-200 rounded-xl')
            ui.button('Reset', on_click=reset_chat).props('no-caps unelevated').classes('w-full h-14 text-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-200 rounded-xl')

    # --- MAIN CHAT ---
    async def send_message():
        text = text_input.value
        if not text: return
        
        text_input.value = ''
        
        # User Message
        with chat_container:
            ui.html(f"<div class='chat-user'>{text}</div>", sanitize=False).classes('w-full flex justify-end mb-4')
        
        app.storage.user['memory']["conversation"].append({"role": "user", "content": text})
        
        # Scroll to bottom
        await ui.run_javascript(f"window.scrollTo(0, document.body.scrollHeight)")
        
        # Loading Indicator
        with chat_container:
            spinner = ui.spinner('dots', size='lg', color='blue-500').classes('ml-4')
        
        # Process Reply
        try:
            reply, expr = await run.io_bound(generate_reply_sync, text, app.storage.user['memory'])
            
            spinner.delete()
            
            # Bot Message
            with chat_container:
                ui.html(f"<div class='chat-bot'>{reply}</div>", sanitize=False).classes('w-full flex justify-start mb-4')
            
            # Update State
            app.storage.user['expression'] = expr
            app.storage.user['memory']["conversation"].append({"role": "assistant", "content": reply})
            await run.io_bound(save_memory, app.storage.user['memory'])
            
            await ui.run_javascript(f"window.scrollTo(0, document.body.scrollHeight)")
            
        except Exception as e:
            spinner.delete()
            ui.notify(f"Error: {str(e)}", color='negative')

    with ui.column().classes('w-full max-w-5xl mx-auto p-4 pb-32 min-h-screen') as chat_container:
        ui.label("Hey Attrangi").classes('text-3xl font-bold text-slate-100')
        ui.label("A safe place to talk.").classes('text-slate-400 mb-8')
        
        # Load History
        for msg in app.storage.user['memory']['conversation']:
            if msg['role'] == 'user':
                ui.html(f"<div class='chat-user'>{msg['content']}</div>", sanitize=False).classes('w-full flex justify-end mb-4')
            else:
                ui.html(f"<div class='chat-bot'>{msg['content']}</div>", sanitize=False).classes('w-full flex justify-start mb-4')

    # --- INPUT AREA ---
    with ui.footer().classes('bg-slate-900 border-t border-slate-800 p-4').style('left: 450px !important; width: calc(100% - 450px) !important;'):
        with ui.row().classes('w-full max-w-5xl mx-auto items-center gap-2 no-wrap'):
            text_input = ui.textarea(placeholder="What's been on your mind?").props('rows=1 rounded outlined bg-color=slate-800 input-class=text-slate-100 placeholder-color=slate-400').classes('w-full text-xl text-slate-100 flex-grow')
            text_input.on('keydown.enter.prevent', send_message)
            
            ui.button(icon='send', on_click=send_message).props('round unelevated').classes('bg-blue-600 hover:bg-blue-500 text-white shadow-md')

import os
ui.run(
    title="Hey Attrangi", 
    dark=True, 
    storage_secret='attrangi_secret_key', 
    port=int(os.environ.get('PORT', 7860)),
    host='0.0.0.0'
)
