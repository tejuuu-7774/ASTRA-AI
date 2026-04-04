from app.config import *
from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import ingest_text
from app.rag import query_rag
from fastapi import UploadFile, File
import shutil
from app.pdf_utils import extract_text_from_pdf

app = FastAPI()

class TextInput(BaseModel):
    text: str

class QuestionInput(BaseModel):
    question: str


# Check if system is alive -> just a test route i added
@app.get("/")
def home():
    return {"message": "AstraAI is now running"}

# Feeds data into memory
@app.post("/ingest")
def ingest(data: TextInput):
    chunks = ingest_text(data.text)
    return {"chunks_created": chunks}

# can ask questions from that memory
@app.post("/ask")
def ask(data: QuestionInput):
    result = query_rag(data.question)
    return result

# This endpoint will accept PDF → read → store → ready for questions
@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text_from_pdf(file_path)

    # Ingest into vector DB
    chunks = ingest_text(text)

    return {
        "message": "PDF processed successfully",
        "chunks_created": chunks
    }