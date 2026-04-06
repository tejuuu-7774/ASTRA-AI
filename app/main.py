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

# Global variable to hold the current vector store instance
vector_db = None

class QuestionInput(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "AstraAI is now running"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db
    
    # CRITICAL: Force clear the previous reference immediately
    vector_db = None
    print("♻️ Resetting Vector DB for new upload...")

    # Read the file bytes
    pdf_bytes = await file.read()
    
    # Extract text from bytes
    text = extract_text_from_pdf_bytes(pdf_bytes)
    
    # DEBUG: Check what was actually extracted in your Render Logs
    print(f"📊 Extracted Text Length: {len(text)} characters")
    if len(text) > 100:
        print(f"📝 Preview of extracted text: {text[:150]}...")
    else:
        print("⚠️ Warning: Extracted text is very short or empty!")

    # Ingest the new text into a fresh, unique collection
    vector_db, chunk_count = ingest_text(text)
    
    print(f"✅ Ingestion complete. Created {chunk_count} chunks.")

    return {
        "message": "PDF processed successfully",
        "chunks_created": chunk_count,
        "preview": text[:50] # Sending a small preview back to UI for verification
    }

@app.post("/ask")
def ask(data: QuestionInput):
    global vector_db

    # Log the incoming question
    print(f"❓ Question received: {data.question}")

    if vector_db is None:
        print("❌ Error: Attempted to ask without an uploaded PDF.")
        return {
            "answer": "Please upload a PDF first.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "No Session"
        }

    # Pass the query to the RAG logic
    response = query_rag(data.question, vector_db)
    
    # Ensure intent_used is always present for the Streamlit UI
    if "intent_used" not in response:
        response["intent_used"] = data.question
        
    return response