from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


# -----------------------------
# Security
# -----------------------------
def is_sensitive_output(text: str):
    sensitive_words = ["password", "secret", "key"]
    return any(word in text.lower() for word in sensitive_words)


# -----------------------------
# Main RAG
# -----------------------------
def query_rag(question: str, vector_db):

    if vector_db is None:
        return {
            "answer": "Please upload a PDF first.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "No DB"
        }

    # STEP 1: Vector Retrieval (semantic)
    docs = vector_db.similarity_search(question, k=5)

    # STEP 2: Keyword Fallback (general, NOT hardcoded)
    try:
        raw_docs = vector_db._collection.get()["documents"]
    except:
        raw_docs = []

    keyword_hits = []
    question_words = [w.lower() for w in question.split() if len(w) > 2]

    for doc_text in raw_docs:
        if any(word in doc_text.lower() for word in question_words):
            keyword_hits.append(doc_text)

    # Convert to doc-like objects
    class Doc:
        def __init__(self, text):
            self.page_content = text

    keyword_docs = [Doc(text) for text in keyword_hits[:3]]

    #  STEP 3: Combine + Deduplicate
    all_docs = docs + keyword_docs

    seen = set()
    unique_docs = []

    for doc in all_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    if not unique_docs:
        return {
            "answer": "Not available in the document.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": question
        }

    # STEP 4: Controlled Context (prevents 413 error)
    MAX_CONTEXT_CHARS = 1800

    context = ""
    used_docs = []

    for doc in unique_docs:
        if len(context) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        context += doc.page_content + "\n"
        used_docs.append(doc)

    sources = list(set([doc.page_content[:100] for doc in used_docs]))

    # STEP 5: LLM Answering
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    You are an intelligent assistant.

    Rules:
    - Use ONLY the provided context
    - Extract the answer clearly if present
    - If partially present, summarize relevant info
    - If not found, say "Not available in the document"
    - Do NOT guess or add external knowledge

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    # STEP 6: Safety Check
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
        "confidence": "MEDIUM",
        "intent_used": question
    }