from app.config import *
from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import ingest_text
from app.rag import query_rag

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