from groq import Groq
from config import GROQ_MODEL, GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def generate_report(memory):
    prompt = f"""
Create a structured mental health reflection report with sections:
1. Key Summary
2. Medical History
3. Psychiatric History
4. Family & Social Background
5. Strengths
6. Screening Impressions (non-diagnostic)
7. Core Issues Summary
8. Goals
9. Recommendations
10. Risk Assessment

User data:
{memory}

Rules:
- Non-diagnostic
- Human tone
- Clear headings
"""

    res = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content
