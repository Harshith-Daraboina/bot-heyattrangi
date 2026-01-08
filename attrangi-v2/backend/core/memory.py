from typing import Dict, Any, List
import uuid
from datetime import datetime

# In-memory session store
# Structure:
# {
#   session_id: {
#     "conversation": [{"role": "user", "content": "..."}],
#     "signals": {},
#     "stage": "exploration",
#     "last_active": timestamp
#   }
# }
SESSIONS: Dict[str, Any] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "conversation": [],
            "signals": {
                "stress": 0,
                "fatigue": 0,
                "low_mood": 0,
                "anxiety": 0,
                "sleep_issues": 0,
                "self_worth": 0,
                "attention": 0
            },
            "stage": "opening",
            "last_active": datetime.now()
        }
    SESSIONS[session_id]["last_active"] = datetime.now()
    return SESSIONS[session_id]

def update_session(session_id: str, data: Dict[str, Any]):
    if session_id in SESSIONS:
        SESSIONS[session_id].update(data)
        SESSIONS[session_id]["last_active"] = datetime.now()

def clear_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]

def add_message(session_id: str, role: str, content: str):
    session = get_session(session_id)
    session["conversation"].append({"role": role, "content": content})
