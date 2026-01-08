from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import uuid
import uvicorn
import os

from core.memory import get_session, update_session, clear_session, add_message
from core.neuro_engine import neuro_engine
from core.retriever import retriever
from core.signals import extract_signals
from core.database import init_db, get_db_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"DB Init failed: {e}")
    yield
    # Shutdown (if needed)

app = FastAPI(title="Attrangi Backend", version="2.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SummaryRequest(BaseModel):
    session_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    
    # 1. Get Session
    session = get_session(session_id)
    
    # 2. Extract Signals and merge with existing
    new_signals = extract_signals(user_message)
    current_signals = session.get("signals", {
        "stress": 0,
        "fatigue": 0,
        "low_mood": 0,
        "anxiety": 0,
        "sleep_issues": 0,
        "self_worth": 0,
        "attention": 0
    })
    
    # Merge signals (accumulate with decay)
    DECAY_FACTOR = 0.8
    for key in current_signals:
        current_signals[key] = int(current_signals[key] * DECAY_FACTOR)
        current_signals[key] += new_signals.get(key, 0)
    
    # Update stage based on signals
    signal_score = sum(current_signals.values())
    if signal_score >= 5:
        stage = "synthesis"
    elif signal_score >= 2:
        stage = "exploration"
    else:
        stage = "opening"
    
    update_session(session_id, {"signals": current_signals, "stage": stage})
    session = get_session(session_id)  # Refresh session with updated stage
    
    # 3. Retrieve Context
    # "YES, if you dump all embeddings into the prompt ❌. NO, if you retrieve top-k relevant chunks ✅"
    # "Use chunks as background guidance. Never quote. Never exceed ~800–1200 tokens."
    context_chunks = retriever.retrieve(user_message, top_k=3) if len(user_message.split()) > 3 else []
    
    # 4. Update Memory (User message -> DB)
    add_message(session_id, "user", user_message)

    # 5. Generate Response
    bot_response = neuro_engine.generate_response(
        message=user_message,
        context=context_chunks,
        session_state=session
    )
    
    # Handle response logic
    reply = bot_response.get("reply") if isinstance(bot_response, dict) else bot_response.reply
    expression = bot_response.get("expression") if isinstance(bot_response, dict) else bot_response.expression
    
    add_message(session_id, "assistant", reply)
    
    return {
        "reply": reply,
        "expression": expression
    }

@app.post("/summary")
async def summary_endpoint(request: SummaryRequest):
    session_id = request.session_id
    session = get_session(session_id)
    
    conversation = session.get("conversation", [])
    if not conversation:
        return {"status": "No conversation to summarize"}
        
    # Generate Summary
    summary_text = neuro_engine.generate_summary(conversation)
    
    # SAVE to DB
    # "SAVES conversation + summary to DB"
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Ensure session exists in DB (if not already insert it? We didn't insert it at start)
        # We need to insert the session first.
        # Check if session exists, if not insert.
        cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO sessions (id) VALUES (%s)", (session_id,))
            
        # Insert Messages
        # Batch insert for efficiency
        args_str = ','.join(cur.mogrify("(%s, %s, %s, %s)", (str(uuid.uuid4()), session_id, m['role'], m['content'])).decode('utf-8') for m in conversation)
        if args_str:
            cur.execute("INSERT INTO messages (id, session_id, role, content) VALUES " + args_str)
        
        # Insert Summary
        cur.execute("INSERT INTO summaries (session_id, summary) VALUES (%s, %s)", (session_id, summary_text))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"status": "success", "summary": summary_text}
        
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_endpoint(request: ChatRequest = Body(...)): 
    # The user request example showed just POST /reset with body likely containing session_id
    # But often reset is just a signal.
    # User request:
    # POST /reset
    # Behavior: Clears in-memory session. Does NOT delete DB data.
    
    # Assuming we need session_id to know WHICH session to reset.
    # Using ChatRequest or customized model. Let's assume input matches request for session_id.
    # If the user meant global reset, that's dangerous. Let's assume session-specific.
    
    session_id = request.session_id
    clear_session(session_id)
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
