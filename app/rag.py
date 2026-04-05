from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


# -----------------------------
# Planner
# -----------------------------
def plan_question(question: str):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    Convert the question into a short search query.
    Keep important keywords. Do not remove meaning.

    Question: {question}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    intent = response.content.strip()
    intent = intent.replace('"', '').replace("'", "")
    return intent


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

    def sanitize_question(q: str):
        for word in ["password", "secret", "key"]:
            q = q.replace(word, "")
        return q.strip()

    clean_question = sanitize_question(question)

    # planner + original query
    intent = plan_question(clean_question if clean_question else question)

    # dual retrieval 
    docs_query = vector_db.similarity_search(question, k=3)
    docs_intent = vector_db.similarity_search(intent, k=3)

    docs = docs_query + docs_intent

    if not docs:
        return {
            "answer": "Not available in the document.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": intent
        }

    # removes duplicates
    unique_docs = []
    seen = set()

    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    # context control (prevents 413 error)
    MAX_CONTEXT_CHARS = 1200

    context = ""
    for doc in unique_docs:
        if len(context) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        context += doc.page_content + "\n"

    sources = list(set([doc.page_content[:100] for doc in unique_docs]))

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    Answer ONLY from the context.
    If not found, say "Not available in the document."

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
        "intent_used": intent
    }