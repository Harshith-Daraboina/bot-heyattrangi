import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"Testing API Key: {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")

if not api_key:
    print("Error: No API Key found.")
    exit(1)

client = Groq(api_key=api_key)

try:
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="llama-3.3-70b-versatile",
    )
    print("SUCCESS: API Key is valid.")
    print("Response:", completion.choices[0].message.content)
except Exception as e:
    print("FAILURE: API Key is invalid or error occurred.")
    print(e)
