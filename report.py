from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from pdf_retriever import pdf_retriever

client = Groq(api_key=GROQ_API_KEY)

REPORT_SYSTEM_PROMPT = """
You are an expert clinical summarizer.
Your goal is to analyze the conversation history and generate a structured clinical report.

You MUST include the following sections. If information is missing for a section, write "Not discussed".

1. Key Summary
2. Medical History
3. Psychiatric History
4. Family & Social Background
5. Strengths
6. Diagnosis (Professional Impression)
7. Assessments (Mention any clear symptoms/signals observed)
8. Core Issues Summary
9. Goals
10. Wider Recommendation (Therapeutic suggestions)
11. Risk Assessment (Self-harm/Suicide indications)
12. Review (Next steps)

Format the output clearly with Markdown headers.
Be objective, professional, and empathetic.
"""

def generate_report(memory):
    conversation = memory["conversation"]
    signals = memory["signals"]
    
    # Create a context string from the last N interactions to avoid token limits if very long,
    # but for now we try to pass most of it.
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])

    # Retrieve PDF context for report
    themes = ", ".join(k for k, v in memory["signals"].items() if v > 1)
    pdf_context = pdf_retriever.retrieve(themes)
    background_knowledge = ""
    if pdf_context:
        background_knowledge = f"\nRelevant Background:\n{' '.join(pdf_context)}\n"
    
    messages = [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation Log:\n{conversation_text}\n\nDetected Signals: {signals}\n{background_knowledge}\n\nPlease generate the comprehensive report."}
    ]

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3 # Lower temperature for factual extraction
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {e}"
