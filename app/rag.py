from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


# -----------------------------
# Security
# -----------------------------
def is_sensitive_output(text: str):
    sensitive_words = ["password", "deeplearn2026"]
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

    # SIMPLE + STRONG RETRIEVAL (NO PLANNER)
    docs = vector_db.similarity_search(question, k=5)

    if not docs:
        return {
            "answer": "Not available in the document.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "No Results"
        }

    # CONTROL CONTEXT (prevents token error)
    MAX_CONTEXT_CHARS = 1800

    context = ""
    used_docs = []

    for doc in docs:
        if len(context) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        context += doc.page_content + "\n"
        used_docs.append(doc)

    sources = list(set([doc.page_content[:100] for doc in used_docs]))

    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    You are an intelligent assistant.

    Rules:
    - Answer using ONLY the provided context
    - If the answer is clearly present, return it directly
    - If partially present, summarize relevant information
    - If not found, say "Not available in the document."
    - Do NOT guess or add external knowledge

    Context:
    {context}

    Question:
    {question}

    Answer: ...
    Confidence: HIGH / MEDIUM / LOW
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    output = response.content

    answer = output
    confidence = "MEDIUM"

    if "Confidence:" in output:
        parts = output.split("Confidence:")
        answer = parts[0].replace("Answer:", "").strip()
        confidence = parts[1].strip()

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
        "confidence": confidence,
        "intent_used": question
    }