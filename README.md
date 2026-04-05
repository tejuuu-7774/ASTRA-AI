# AstraAI

AstraAI is a Retrieval-Augmented Generation (RAG) system that answers questions from uploaded PDFs.
It ensures responses are **strictly based on document context**, avoiding hallucinations.

---

## Live Demo

Frontend: https://astra-ai-ui.onrender.com
Backend: https://astra-ai-q72g.onrender.com

---

## Overview

AstraAI improves reliability in AI responses by combining retrieval with generation.

Instead of guessing, it:

* Retrieves relevant document content
* Uses it to generate grounded answers

**Core capabilities:**

* Context-based answering
* Query refinement (agentic planning)
* Confidence scoring
* Sensitive data filtering

---

## Tech Stack

**Frontend**

* Streamlit
* Requests

**Backend**

* FastAPI
* Uvicorn

**AI / RAG**

* LangChain
* ChromaDB (in-memory vector store)
* HuggingFace API (embeddings)
* Groq (LLaMA 3.1)

**Utilities**

* PyPDF
* Python Dotenv

---

## How It Works

1. Upload PDF
2. Extract and split text into chunks
3. Generate embeddings and store in an in-memory ChromaDB vector store

When asking a question:

1. Refine query (planner)
2. Retrieve relevant chunks
3. Generate answer from context
4. Return answer + confidence + sources

---

## Features

* PDF upload (no file storage)
* Semantic search with vector DB
* Context-grounded answers only
* Confidence levels (HIGH / MEDIUM / LOW)
* Intent-based query processing
* Sensitive data protection
* Fully deployed system

---

## Project Structure

```
ASTRA-AI/
│
├── app/
│   ├── main.py
│   ├── ingest.py
│   ├── rag.py
│   ├── embeddings.py
│   ├── pdf_utils.py
│
├── ui.py
├── requirements.txt
├── .env
└── assets/
```

---

## API Endpoints

* GET `/` → Health check
* POST `/ingest` → Ingest text
* POST `/upload_pdf` → Upload PDF
* POST `/ask` → Ask question

---

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

## Local Setup

Clone repository:

```
git clone https://github.com/tejuuu-7774/ASTRA-AI.git
cd ASTRA-AI
```

Install dependencies:

```
pip install -r requirements.txt
```

Run backend:

```
uvicorn app.main:app --reload
```

Run frontend:

```
streamlit run ui.py
```

---

## Key Decisions

* Used HuggingFace API to reduce memory usage
* Used Groq for fast inference
* Optimized for free-tier deployment
* Processed PDFs in-memory (no storage)

---

## Limitations

* Single document at a time
* No authentication
* No persistent memory
* No streaming responses
* Data is stored in-memory; users must re-upload PDFs after server restarts

---

## Future Improvements

* Multi-document support
* User sessions
* Better UI/UX
* Streaming responses
* Authentication

---

## Author

Tejaswini Palwai

---

## Note
AstraAI focuses on **accuracy over creativity**.
It is built to answer only what it knows—and say nothing when it doesn’t.
