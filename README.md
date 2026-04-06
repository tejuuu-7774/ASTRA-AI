# AstraAI

AstraAI is a Retrieval-Augmented Generation (RAG) system that answers questions from uploaded PDFs.  
It ensures responses are **strictly grounded in document context**, eliminating hallucinations.

---

## Live Demo

Frontend: https://astra-ai-ui.onrender.com  
Backend: https://astra-ai-q72g.onrender.com  

---

## Overview

AstraAI improves reliability in AI systems by combining retrieval with generation.

Instead of guessing, it:
- Retrieves relevant document content  
- Generates answers only from that context  

**Core capabilities:**
- Context-grounded answering  
- Hybrid retrieval (semantic + keyword fallback)  
- Confidence scoring  
- Optional security filtering for sensitive data  

---

## Tech Stack

**Frontend**
- Streamlit  
- Requests  

**Backend**
- FastAPI  
- Uvicorn  

**AI / RAG**
- LangChain  
- ChromaDB (in-memory vector store)  
- HuggingFace Inference API (embeddings)  
- Groq (LLaMA 3.1)  

**Utilities**
- PyMuPDF (robust PDF parsing)  
- Python Dotenv  

---

## How It Works

1. Upload a PDF  
2. Extract and split text into chunks  
3. Generate embeddings and store in an in-memory vector store  

When querying:
1. Process and refine the question  
2. Retrieve relevant chunks (semantic + keyword boost)  
3. Build a controlled context window  
4. Generate answer using LLM  
5. Return answer, confidence, and sources  

---

## Features

- PDF upload with in-memory processing (no file storage)  
- Hybrid retrieval for improved accuracy  
- Context-only answers (no hallucination)  
- Confidence classification (HIGH / MEDIUM / LOW)  
- Source transparency  
- Optional sensitive data filtering  
- Fully deployed frontend and backend  

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

- `GET /` → Health check  
- `POST /upload_pdf` → Upload and process PDF  
- `POST /ask` → Query document  

---

## Environment Variables

```

GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token

```

---

## Local Setup

Clone the repository:

```

git clone [https://github.com/tejuuu-7774/ASTRA-AI.git](https://github.com/tejuuu-7774/ASTRA-AI.git)
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

## Key Engineering Decisions

- Used HuggingFace API to avoid heavy local models and reduce memory usage  
- Integrated Groq for fast and efficient inference  
- Implemented hybrid retrieval (semantic + keyword) to improve accuracy on structured queries  
- Used PyMuPDF for reliable extraction from complex PDF layouts  
- Designed system to operate within free-tier deployment constraints  

---

## Limitations

- Supports a single document at a time  
- No authentication or user sessions  
- No persistent storage (data resets on restart)  
- No streaming responses  

---

## Future Improvements

- Multi-document indexing  
- User-based session memory  
- Streaming responses  
- Authentication and access control  
- Enhanced UI/UX  

---

## Author

Tejaswini Palwai  

---

## Note

AstraAI prioritizes **accuracy over creativity**.  
It answers only what exists in the document—and refuses to guess.

