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


def analyze_email(subject: str, body: str) -> dict:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are an AI Email Operations Assistant for businesses.

Analyze the email and return ONLY valid JSON in this format:

{
  "categories": [],
  "priority": "low | medium | high",
  "sentiment": "neutral | frustrated | angry | positive",
  "entities": {},
  "route_to": [],
  "actions": [],
  "confidence": 0.0,
  "requires_human_review": false,
  "decision_summary": "",
  "suggested_reply": ""
}

Allowed categories:
refund, complaint, appointment_request, general_inquiry

Allowed entity keys:
name, email, phone, order_id, date, time, company, product

Allowed route_to values:
support, billing, scheduling, sales, general_support

Rules:
- categories must be a list.
- Include all clearly present categories.
- Use only allowed categories.
- If no specific category fits, use ["general_inquiry"].
- entities must include only keys found in the email.
- actions must be clear business next steps.
- confidence must be a number between 0 and 1.
- requires_human_review must be true if the email is angry, unclear, high priority, or contains multiple categories.
- decision_summary must be one short sentence.
- suggested_reply must address all categories in one professional reply.
- Return no text outside JSON.
"""
            },
            {
                "role": "user",
                "content": f"Subject: {subject}\nBody: {body}"
            }
        ],
        text={"format": {"type": "json_object"}},
        temperature=0
    )

    return json.loads(response.output_text)


@app.post("/classify-email")
def classify_email(payload: EmailIn):

    full_text = f"Subject: {payload.subject}\nBody: {payload.body}"

    if check_moderation(full_text):
        return {"error": "Content violates safety policies."}

    result = analyze_email(payload.subject, payload.body)

    allowed_categories = ["refund", "complaint", "appointment_request", "general_inquiry"]
    allowed_priorities = ["low", "medium", "high"]
    allowed_sentiments = ["neutral", "frustrated", "angry", "positive"]
    allowed_routes = ["support", "billing", "scheduling", "sales", "general_support"]

    if not isinstance(result.get("categories"), list):
        return {"error": "categories must be a list"}

    for category in result["categories"]:
        if category not in allowed_categories:
            return {"error": "Invalid category", "value": category}

    if result.get("priority") not in allowed_priorities:
        return {"error": "Invalid priority", "value": result.get("priority")}

    if result.get("sentiment") not in allowed_sentiments:
        return {"error": "Invalid sentiment", "value": result.get("sentiment")}

    if not isinstance(result.get("entities"), dict):
        return {"error": "entities must be a dictionary"}

    if not isinstance(result.get("route_to"), list):
        return {"error": "route_to must be a list"}

    for route in result["route_to"]:
        if route not in allowed_routes:
            return {"error": "Invalid route", "value": route}

    if not isinstance(result.get("actions"), list):
        return {"error": "actions must be a list"}

    if not isinstance(result.get("confidence"), (int, float)):
        return {"error": "confidence must be a number"}

    if not isinstance(result.get("requires_human_review"), bool):
        return {"error": "requires_human_review must be true or false"}

    return result

@app.get("/health")
def health():
    return {"status": "ok"}