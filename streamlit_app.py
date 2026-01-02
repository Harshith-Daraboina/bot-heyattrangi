import streamlit as st
from pathlib import Path
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from memory import load_memory, save_memory, reset_memory
from signals import extract_signals
from report import generate_report
import pdf_retriever as pdf_retriever_module
import pdf_embedder
import importlib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Hey Attrangi",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# STYLING
# ==============================
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');

    /* GLOBAL RESET & FONT */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* MAIN BACKGROUND - Deep Blue Slate */
    .stApp {
        background-color: #0F172A; /* Slate 900 */
        color: #F1F5F9; /* Slate 100 */
    }

    /* SIDEBAR BACKGROUND */
    [data-testid="stSidebar"] {
        background-color: #020617; /* Slate 950 (Darker) */
        border-right: 1px solid #1E293B; /* Slate 800 */
    }

    /* HEADERS */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600;
    }
    
    p, span, div, label {
        color: #CBD5E1; /* Slate 300 */
    }

    /* BUTTONS */
    .stButton button {
        background-color: #1E293B; /* Slate 800 */
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 12px;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #334155;
        border-color: #60A5FA; /* Blue 400 */
        color: #60A5FA;
    }

    /* CHAT INPUT CONTAINER - FIXED TO DARK */
    .stChatInputContainer {
        background-color: #0F172A; /* Match Main BG */
        border-top: 1px solid #1E293B;
        padding-bottom: 3rem;
    }
    
    /* Input Text Area */
    div[data-baseweb="textarea"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #F8FAFC;
    }
    
    div[data-baseweb="textarea"] textarea {
        color: #F8FAFC !important;
        caret-color: #60A5FA;
    }
    
    /* Placeholder Color */
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #64748B;
    }

    /* CHAT BUBBLES */
    .stChatMessage {
        padding-bottom: 1rem;
        background: transparent;
    }

    /* User (Right) - Blue Tint */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        flex-direction: row-reverse !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background-color: #1E3A8A; /* Blue 900 */
        border: 1px solid #2563EB; /* Blue 600 */
        color: #F8FAFC !important;
        border-radius: 16px 16px 4px 16px;
        padding: 1rem;
    }

    /* Bot (Left) - Slate Gray */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background-color: #1E293B; /* Slate 800 */
        border: 1px solid #334155; /* Slate 700 */
        color: #E2E8F0 !important;
        border-radius: 16px 16px 16px 4px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# EMBEDDINGS INIT
# ==============================
VECTOR_PATH = Path("vector_store/mental_health.index")

@st.cache_resource
def get_retriever():
    if not VECTOR_PATH.exists():
        with st.spinner("Building knowledge base…"):
            pdf_embedder.build_index()
    importlib.reload(pdf_retriever_module)
    return pdf_retriever_module.pdf_retriever

retriever = get_retriever()

# ==============================
# CLIENT
# ==============================
client = Groq(api_key=GROQ_API_KEY)

# ==============================
# SYSTEM PROMPT
# ==============================
SYSTEM_PROMPT = """
You are Attrangi — a warm, emotionally intelligent mental health companion.

Your goals:
- Make the user feel understood
- Explore what matters most
- Avoid repetition
- Ask at most ONE open-ended question when helpful
- Stop questioning when enough context is clear
- Never reintroduce yourself
- Never loop generic reassurance

Conversation stages:
- opening → exploration → synthesis

End every reply with exactly:
[EXPRESSION: EMPATHETIC | STRESSED | TIRED | REFLECTIVE | NEUTRAL | SAFETY]
"""

EXPRESSION_MAP = {
    "EMPATHETIC": "public/bot_expressions/EMPATHETIC.jpg",
    "STRESSED": "public/bot_expressions/STRESSED.jpg",
    "TIRED": "public/bot_expressions/TIRED.jpg",
    "REFLECTIVE": "public/bot_expressions/REFLECTIVE.jpg",
    "NEUTRAL": "public/bot_expressions/NEUTRAL.jpg",
    "SAFETY": "public/bot_expressions/SAFETY.jpg",
}

# ==============================
# HELPERS
# ==============================
def update_stage(memory):
    score = sum(memory["signals"].values())
    if score >= 5:
        memory["stage"] = "synthesis"
    elif score >= 2:
        memory["stage"] = "exploration"
    else:
        memory["stage"] = "opening"

def generate_reply(user_text, memory):
    extract_signals(user_text, memory)
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

    messages.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.85
    )

    full = completion.choices[0].message.content

    import re
    match = re.search(r"\[EXPRESSION:\s*([A-Z]+)\]", full)
    expression = match.group(1) if match else "NEUTRAL"
    clean = re.sub(r"\[EXPRESSION:.*?\]", "", full).strip()

    return clean, expression

# ==============================
# SESSION STATE
# ==============================
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

if "expression" not in st.session_state:
    st.session_state.expression = "NEUTRAL"

# ==============================
# LAYOUT
# ==============================
# ==============================
# LAYOUT
# ==============================
with st.sidebar:
    st.markdown("## Attrangi")
    st.image(
        EXPRESSION_MAP.get(st.session_state.expression),
        use_container_width=True
    )

    if st.button("🧾 Generate Summary"):
        report = generate_report(st.session_state.memory)
        st.text_area("Mental Health Summary", report, height=350)

    if st.button("🔄 Reset"):
        st.session_state.memory = reset_memory()
        st.session_state.expression = "NEUTRAL"
        st.rerun()

# Main Chat Area
st.markdown("## 🧠 Hey Attrangi")
st.markdown("A safe place to talk.")

for msg in st.session_state.memory["conversation"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# CHAT INPUT
# ==============================
if user_input := st.chat_input("What's been on your mind?"):
    st.session_state.memory["conversation"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply, expr = generate_reply(
                user_input, st.session_state.memory
            )
            st.markdown(reply)

    st.session_state.expression = expr
    st.session_state.memory["conversation"].append(
        {"role": "assistant", "content": reply}
    )

    save_memory(st.session_state.memory)
    st.rerun()

# ==============================
# AUTO-SCROLL
# ==============================
import streamlit.components.v1 as components
components.html(
    """
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, window.parent.document.querySelector('section.main').scrollHeight);
    </script>
    """,
    height=0
)
