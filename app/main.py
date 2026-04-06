print("🔥 Starting AstraAI backend...")

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingest import ingest_text
from app.rag import query_rag
from app.pdf_utils import extract_text_from_pdf_bytes

app = FastAPI()

vector_db = None


class QuestionInput(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "AstraAI is now running"}


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db

    vector_db = None
    print("♻️ Resetting DB")

    pdf_bytes = await file.read()
    text = extract_text_from_pdf_bytes(pdf_bytes)

    vector_db, chunks = ingest_text(text)

    print(f"✅ Created {chunks} chunks")

    return {
        "message": "PDF processed successfully",
        "chunks_created": chunks
    }


@app.post("/ask")
def ask(data: QuestionInput):
    global vector_db

    if vector_db is None:
        return {
            "answer": "Please upload a PDF first.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "No DB"
        }

    response = query_rag(data.question, vector_db)

    if "intent_used" not in response:
        response["intent_used"] = data.question

    return response