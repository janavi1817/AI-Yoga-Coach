import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Fallback to checking GOOGLE_API_KEY if GEMINI_API_KEY is missing
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key or api_key == "YOUR_API_KEY_HERE":
    print("Error: Valid GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file or environment.")
    exit(1)

# Configure the SDK with the API key
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

name = input("Enter your name: ")
print(f"\nWelcome {name}! Placement Preparation Bot (type 'exit' to stop)\n")

system_prompt = f"""
You are a multilingual placement preparation assistant.

User: {name}

Rules:
- Answer ONLY placement-related questions (aptitude, interview, resume, skills).
- Politely refuse non-placement questions.
- No medical, legal, or psychological advice.
- Keep answers under 80 words.
- Be professional and encouraging.
"""

while True:
    try:
        q = input("You: ")
        if q.lower() == "exit":
            print("Bot: All the best for your placements 🌟")
            break

        if not q.strip():
            continue

        # Combine system prompt with user question
        full_prompt = f"{system_prompt}\nQuestion: {q}"
        
        response = model.generate_content(full_prompt)
        
        if response.text:
            print(f"Bot: {response.text}")
        else:
            print("Bot: I'm sorry, I couldn't generate a response. Please try again.")

    except Exception as e:
        if "quota" in str(e).lower():
            print("Bot: API quota limit reached. Please try again later.")
        else:
            print(f"Bot: An error occurred: {e}")
