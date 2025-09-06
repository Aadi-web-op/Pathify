# server_fastapi/groq_client.py

import os
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_sync_groq(system_prompt: str, user_prompt: str, is_json: bool = False):
    """
    Calls the Groq API synchronously.
    Returns a string (either plain text or a JSON string).
    """
    response_format = {"type": "json_object"} if is_json else None
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            response_format=response_format
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        # Return a JSON string with error info to maintain type consistency
        return json.dumps({"error": "Failed to get AI response.", "details": str(e)})