
from app import generate_reply
from memory import load_memory

# Mock memory
memory = load_memory()

test_inputs = [
    "Hello, how are you?",
    "I'm feeling incredibly overwhelmed and panic is setting in.",
    "I'm just really tired and want to sleep for a week.",
    "I'm thinking about my past choices.",
    "I feel happy today!"
]

print("--- START DEBUG ---")
for text in test_inputs:
    print(f"\nUser: {text}")
    reply, expr = generate_reply(text, memory)
    print(f"Bot: {reply[:50]}...")
    print(f"Expression: {expr}")
print("--- END DEBUG ---")
