from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


# -----------------------------
# Security Check
# -----------------------------
def is_sensitive_output(text: str):
    # Change "key" to specific technical terms to avoid false positives
    sensitive_words = ["password", "secret_key", "api_key", "private_key"]
    return any(word in text.lower() for word in sensitive_words)

def is_sensitive_query(text: str):
    # Do the same for the question check
    sensitive_words = ["password", "wifi password", "secret key", "api key"]
    return any(word in text.lower() for word in sensitive_words)

# -----------------------------
# Main RAG Pipeline
# -----------------------------
def query_rag(question: str, vector_db):

    if is_sensitive_query(question):
        return {
            "answer": "This information cannot be disclosed.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "Blocked"
        }
        
    if vector_db is None:
        return {
            "answer": "Please upload a PDF first.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "No DB"
        }

    # -----------------------------
    # STEP 1: Vector Search
    # -----------------------------
    docs = vector_db.similarity_search(question, k=8)

    # -----------------------------
    # STEP 2: Keyword Rescue (FINAL)
    # -----------------------------
    try:
        all_texts = vector_db._collection.get()["documents"]
    except:
        all_texts = []

    class Doc:
        def __init__(self, text):
            self.page_content = text

    # Extract meaningful words from question
    question_words = [
        w.lower().strip("?!.:,")
        for w in question.split()
        if len(w) > 2
    ]

    # Add generic intent boost (NOT hardcoded to PDF)
    if any(q in question.lower() for q in ["time", "timing", "schedule", "when", "open"]):
        question_words.extend(["time", "timing", "schedule", "hour"])

    keyword_docs = []

    for text in all_texts:
        if any(word in text.lower() for word in question_words):
            keyword_docs.append(Doc(text))

    # -----------------------------
    # STEP 3: Merge (Keyword Priority)
    # -----------------------------
    all_docs = keyword_docs + docs

    unique_docs = []
    seen = set()

    for d in all_docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    # -----------------------------
    # STEP 4: Context Building (Token Safe)
    # -----------------------------
    MAX_CONTEXT = 1800
    context = ""
    used_docs = []

    for d in unique_docs:
        if len(context) + len(d.page_content) > MAX_CONTEXT:
            break
        context += d.page_content + "\n"
        used_docs.append(d)

    if not context.strip():
        return {
            "answer": "Not available in the document.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": question
        }

    sources = list(set([d.page_content[:100] + "..." for d in used_docs]))

    # -----------------------------
    # STEP 5: LLM Answer
    # -----------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
Answer ONLY from the context below.
If the answer is not present, say: Not available in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # -----------------------------
    # STEP 6: Security Filter
    # -----------------------------
    if is_sensitive_output(answer):
        return {
            "answer": "This information cannot be disclosed.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "Blocked"
        }

    return {
        "answer": answer,
        "sources": sources,
        "confidence": "MEDIUM" if len(sources) > 0 else "LOW",
        "intent_used": question
    }