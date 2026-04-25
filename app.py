import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Email Classifier")

class EmailIn(BaseModel):
    subject: str
    body: str

def ask_ai(subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"""
Classify this email into one category:
refund, complaint, appointment_request, general_inquiry

Email:
Subject: {subject}
Body: {body}
"""
    )

    return response.output_text

@app.post("/classify-email")
def classify_email(payload: EmailIn):
    ai_result = ask_ai(payload.subject, payload.body)

    return {
        "ai_response": ai_result
    }

@app.get("/health")
def health():
    return {"status": "ok"}