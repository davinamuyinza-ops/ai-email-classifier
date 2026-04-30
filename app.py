import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel, Field
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Email Classifier")

class EmailIn(BaseModel):
    subject: str = Field(default="", max_length=200)
    body: str = Field(..., min_length=10, max_length=5000)


def check_moderation(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    return result.results[0].flagged

def ask_ai(subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a strict email classification system.

Allowed categories:
refund, complaint, appointment_request, general_inquiry

Return ONLY valid JSON in this format:
{
  "category": "",
  "priority": "low | medium | high",
  "entities": {},
  "suggested_reply": ""
}

Rules:
- category must be EXACTLY one of the allowed values
- do not invent new categories
- do not explain anything
- do not return text outside JSON
- entities must only include keys that appear in the email
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

    full_text = f"Subject: {payload.subject}\nBody: {payload.body}"

    flagged = check_moderation(full_text)

    if flagged:
        return {"error": "Content violates safety policies."}

    ai_result = ask_ai(payload.subject, payload.body)

    try:
        parsed = json.loads(ai_result)
    except Exception:
        return {"error": "Invalid AI response", "raw": ai_result}

    return parsed

@app.get("/health")
def health():
    return {"status": "ok"}