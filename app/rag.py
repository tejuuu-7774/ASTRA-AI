from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


def plan_question(question: str):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    Convert the question into a SHORT search query (5-8 words max).
    No explanation. Only keywords.

    Question: {question}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    intent = response.content.strip()
    intent = intent.replace('"', '').replace("'", "")
    return intent


def is_sensitive_output(text: str):
    sensitive_words = ["password", "deeplearn2026"]
    return any(word in text.lower() for word in sensitive_words)


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
    intent = plan_question(clean_question if clean_question else question)

    # Reduced retrieval
    docs = vector_db.similarity_search(intent, k=2)

    if not docs:
        return {
            "answer": "Not available in the document.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": intent
        }

    # Limit context size (CRITICAL FIX)
    context = "\n".join([doc.page_content[:300] for doc in docs])

    sources = list(set([doc.page_content[:100] for doc in docs]))

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