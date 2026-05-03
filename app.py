import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Email Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailIn(BaseModel):
    subject: str = Field(default="", max_length=200)
    body: str = Field(..., min_length=10, max_length=5000)


def check_moderation(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    return result.results[0].flagged


def classify_step(subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": """
Classify the email into exactly ONE category:

refund, complaint, appointment_request, general_inquiry

Return ONLY the category name.
No explanations.
"""},
            {"role": "user", "content": f"Subject: {subject}\nBody: {body}"}
        ]
    )
    return response.output_text.strip().lower()

def priority_step(subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
Determine the priority of the email.

Return ONLY one of:
low, medium, high

No explanation.
"""
            },
            {
                "role": "user",
                "content": f"Subject: {subject}\nBody: {body}"
            }
        ]
    )
    return response.output_text.strip().lower()

def extract_entities_step(subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
           {"role": "system", "content": """
Extract entities from the email.

Allowed keys:
name, email, phone, order_id, date, time

Return ONLY valid JSON.
Include only keys that exist in the email.
"""},
            {"role": "user", "content": f"Subject: {subject}\nBody: {body}"}
        ],
        text={"format": {"type": "json_object"}}
    )
    return json.loads(response.output_text)

def decision_summary_step(category: str, subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
Write ONE short sentence explaining the classification.

Do not include reasoning steps.
Be concise.
"""
            },
            {
                "role": "user",
                "content": f"Category: {category}\nSubject: {subject}\nBody: {body}"
            }
        ]
    )
    return response.output_text.strip()

def reply_step(category: str, subject: str, body: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": """
Generate a short, professional, helpful reply.

Do not be long.
Be polite and clear.
"""},
            {"role": "user", "content": f"Category: {category}\nSubject: {subject}\nBody: {body}"}
        ]
    )
    return response.output_text



@app.post("/classify-email")
def classify_email(payload: EmailIn):

    full_text = f"Subject: {payload.subject}\nBody: {payload.body}"

    if check_moderation(full_text):
        return {"error": "Content violates safety policies."}

    category = classify_step(payload.subject, payload.body)
    priority = priority_step(payload.subject, payload.body)
    entities = extract_entities_step(payload.subject, payload.body)
    summary = decision_summary_step(category, payload.subject, payload.body)
    reply = reply_step(category, payload.subject, payload.body)

    # Output checking
    allowed_categories = ["refund", "complaint", "appointment_request", "general_inquiry"]
    allowed_priorities = ["low", "medium", "high"]

    if category not in allowed_categories:
        return {"error": "Invalid category", "value": category}

    if priority not in allowed_priorities:
        return {"error": "Invalid priority", "value": priority}

    if not isinstance(entities, dict):
        return {"error": "Entities must be a dictionary"}

    return {
        "category": category,
        "priority": priority,
        "entities": entities,
        "decision_summary": summary,
        "suggested_reply": reply
    }

@app.get("/health")
def health():
    return {"status": "ok"}