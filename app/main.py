from app.config import *
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingest import ingest_text
from app.rag import query_rag
from app.pdf_utils import extract_text_from_pdf_bytes

app = FastAPI()

class TextInput(BaseModel):
    text: str

class QuestionInput(BaseModel):
    question: str


# Check if system is alive
@app.get("/")
def home():
    return {"message": "AstraAI is now running"}


# Feeds raw text into memory
@app.post("/ingest")
def ingest(data: TextInput):
    chunks = ingest_text(data.text)
    return {"chunks_created": chunks}


# Ask questions from stored knowledge
@app.post("/ask")
def ask(data: QuestionInput):
    result = query_rag(data.question)
    return result


# 🔥 UPDATED PDF UPLOAD (NO FILE SAVING)
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    # Read file directly into memory
    pdf_bytes = await file.read()

    # Extract text from memory
    text = extract_text_from_pdf_bytes(pdf_bytes)

    # Ingest into vector DB
    chunks = ingest_text(text)

    return {
        "message": "PDF processed successfully",
        "chunks_created": chunks
    }