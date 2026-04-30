import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Email Classifier")

class EmailIn(BaseModel):
    subject: str
    body: str

def ask_ai(subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are an email classification system.

Return ONLY valid JSON in this format:
{
  "category": "refund | complaint | appointment_request | general_inquiry",
  "priority": "low | medium | high",
  "suggested_reply": "short helpful reply"
}
"""
            },
            {
                "role": "user",
                "content": f"Subject: {subject}\nBody: {body}"
            }
        ],
        text={"format": {"type": "json_object"}}
    )

    return response.output_text


@app.post("/classify-email")
def classify_email(payload: EmailIn):
    ai_result = ask_ai(payload.subject, payload.body)

    try:
        parsed = json.loads(ai_result)
    except Exception:
        return {"error": "Invalid AI response", "raw": ai_result}

    return parsed

@app.get("/health")
def health():
    return {"status": "ok"}