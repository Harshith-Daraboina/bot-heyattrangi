import re

QUESTION_PATTERNS = [
    r"\?$",
    r"\bwhy\b",
    r"\bwhat\b",
    r"\bhow\b",
    r"\btell me\b",
    r"\bwhat is\b",
    r"\bwhat should\b",
]

# States
BOT_LEADS = "bot_leads"
USER_LEADS = "user_leads"

def user_asked_question(text: str) -> bool:
    """
    Determines if the user text contains a question based on hard regex patterns.
    """
    t = text.lower().strip()
    return any(re.search(p, t) for p in QUESTION_PATTERNS)
