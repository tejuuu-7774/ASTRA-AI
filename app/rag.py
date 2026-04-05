from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


def is_sensitive_output(text: str):
    sensitive_words = ["password", "deeplearn2026"]
    return any(word in text.lower() for word in sensitive_words)


def query_rag(question: str, vector_db):

    # Simple + reliable retrieval
    docs = vector_db.similarity_search(question, k=5)

    if not docs:
        return {
            "answer": "Not available in the document.",
            "sources": [],
            "confidence": "LOW"
        }

    # Controlled context (prevents token error)
    MAX_CONTEXT_CHARS = 1800

    context = ""
    used_docs = []

    for doc in docs:
        if len(context) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        context += doc.page_content + "\n"
        used_docs.append(doc)

    sources = list(set([doc.page_content[:100] for doc in used_docs]))

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    You are an intelligent assistant.

    Rules:
    - Use ONLY the given context
    - If answer exists, extract it clearly
    - If not present, say "Not available in the document"
    - Do NOT guess

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    if is_sensitive_output(answer):
        return {
            "answer": "This information cannot be disclosed.",
            "sources": [],
            "confidence": "LOW"
        }

    return {
        "answer": answer,
        "sources": sources,
        "confidence": "MEDIUM"
    }