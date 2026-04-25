from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Email Classifier")

class EmailIn(BaseModel):
    subject: str
    body: str


@app.post("/classify-email")
def classify_email(payload: EmailIn):
    return {
        "received_subject": payload.subject,
        "received_body": payload.body
    }

@app.get("/health")
def health():
    return {"status": "ok"}