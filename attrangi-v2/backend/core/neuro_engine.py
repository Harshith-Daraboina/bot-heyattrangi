import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class BotResponse(BaseModel):
    reply: str = Field(description="The response to the user")
    expression: str = Field(description="The facial expression of the bot: EMPATHETIC, STRESSED, TIRED, REFLECTIVE, SAFETY, or NEUTRAL")

SYSTEM_PROMPT = """You are a warm, emotionally intelligent mental health companion.

Your role is to help the user feel heard and gently understand what they are going through.
You are having a real human conversation, not conducting therapy or an interview.

────────────────────────
Core Style Rules
────────────────────────
- Respond like a real person, not a therapist or chatbot
- Use natural, everyday language
- Do NOT use generic empathy phrases such as:
  "That can be really tough"
  "I'm sorry you're going through this"
  "It sounds like…"
- Vary sentence structure and emotional framing each turn
- Never interrogate, rush, or overwhelm the user
- Do NOT reintroduce yourself once the conversation has started
- If the user says "hi" again mid-conversation, treat it as continuation, not a restart

IMPORTANT:
You must never explain concepts, symptoms, or psychology unless the user explicitly asks.
When emotions are present, prioritize presence over explanation.

You must not repeat the same emotional acknowledgment twice in a row.
If a similar emotion appears again, respond differently.

────────────────────────
How to Respond
────────────────────────
1. Acknowledge the user's situation or feeling in a specific, human way  
   (reflect *their* words, not a template)

2. Choose ONE of the following paths:
   - **Soft invitation (no question)** when emotions are heavy  
     Example: "We can sit with this for a moment if you want."
   - **One targeted, open-ended question** when exploration helps
   - **Reflection only** when the user is already sharing deeply

3. If the user's input is short or vague, ask ONE clarifying question  
   (avoid "Tell me more")

────────────────────────
Question Guidance
────────────────────────
When asking a question, explore only ONE dimension:
- Cause → "What do you think set this off?"
- Impact → "What's been hardest day to day?"
- Meaning → "What has this made you question?"
- Attachment → "What do you miss, or don't miss?"

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
Choose ONE expression based on the user's current state:
- EMPATHETIC → sadness, grief, emotional pain
- STRESSED → anxiety, overwhelm, pressure
- TIRED → exhaustion, low energy, sleep difficulty
- REFLECTIVE → thinking aloud, meaning-making
- SAFETY → explicit self-harm or severe crisis
- NEUTRAL → greetings or low emotional intensity

────────────────────────
────────────────────────
Output Format
────────────────────────
Respond naturally as a human. Do NOT output JSON.
At the end of your response, on a new line, you MUST append one of these tags:

[EXPRESSION: EMPATHETIC]
[EXPRESSION: STRESSED]
[EXPRESSION: TIRED]
[EXPRESSION: REFLECTIVE]
[EXPRESSION: SAFETY]
[EXPRESSION: NEUTRAL]

Example:
I hear you, and it makes sense properly.
[EXPRESSION: EMPATHETIC]
"""

class NeuroEngine:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")
        
        self.llm = ChatGroq(
            temperature=0.85,  # Higher temperature for more natural responses
            model_name="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            max_tokens=350  # Limit response length
        )

    def _compress_context(self, chunks, max_chars=600):
        """Compress context to prevent token bloat"""
        joined = " ".join(chunks)
        return joined[:max_chars]
    
    def _get_recent_conversation(self, conversation, limit=6):
        """Get recent conversation history"""
        return conversation[-limit:] if len(conversation) > limit else conversation
    
    def _calculate_stage(self, signals):
        """Calculate conversation stage based on signal strength"""
        if isinstance(signals, dict):
            score = sum(signals.values())
        else:
            score = 0
            
        if score >= 5:
            return "synthesis"
        elif score >= 2:
            return "exploration"
        else:
            return "opening"

    def generate_response(self, message: str, context: list, session_state: dict):
        try:
            # Get recent conversation
            conversation = session_state.get("conversation", [])
            recent = self._get_recent_conversation(conversation)
            
            # Calculate stage
            signals = session_state.get("signals", {})
            stage = self._calculate_stage(signals)
            
            # Build messages directly using LangChain message classes (avoids template variable issues)
            langchain_messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(content=f"Conversation stage: {stage}"),
                SystemMessage(content=f"Recent conversation: {recent}"),
            ]
            
            # Add context if available (compressed)
            if context and len(message.split()) > 3:  # Skip for very short inputs
                compressed_context = self._compress_context(context)
                langchain_messages.append(SystemMessage(
                    content=(
                        "Internal reference material (DO NOT quote, summarize, or explain directly).\n"
                        "Use only to guide tone, emotional pacing, and choice of questions.\n\n"
                        + compressed_context
                    )
                ))
            
            # Add dynamic instructions
            langchain_messages.append(SystemMessage(
                content="If the user names a new emotion, respond in a new way."
            ))
            
            if stage == 'opening' and len(recent) == 0:
                langchain_messages.append(SystemMessage(
                    content="This is the start. Vary your greeting. Do NOT simply say 'It's nice to meet you'."
                ))
            
            langchain_messages.append(HumanMessage(content=message))
            
            # Invoke LLM directly with messages
            llm_response = self.llm.invoke(langchain_messages)
            response_text = llm_response.content.strip()
            
            # Parse Text + Tag
            import re
            
            # Default values
            reply = response_text
            expression = "NEUTRAL"
            
            # Regex to find [EXPRESSION: TAG]
            match = re.search(r'\[EXPRESSION:\s*([A-Z_]+)\]', response_text)
            if match:
                expression = match.group(1)
                # Remove the tag from the reply
                reply = response_text.replace(match.group(0), "").strip()
            
            return {
                "reply": reply,
                "expression": expression
            }
        except Exception as e:
            print(f"Error in NeuroEngine: {e}")
            import traceback
            traceback.print_exc()
            return {
                "reply": "I'm having a little trouble thinking right now, but I'm here for you.",
                "expression": "NEUTRAL"
            }
            
    def generate_summary(self, conversation_history: list):
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the following conversation and generate a clinical summary."),
            ("human", "Conversation:\n{conversation}")
        ])
        summary_chain = summary_prompt | self.llm
        
        conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
        
        res = summary_chain.invoke({"conversation": conv_text})
        return res.content

neuro_engine = NeuroEngine()