from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.embeddings import get_embeddings
import os


# -----------------------------
# Planner (Agent Layer)
# -----------------------------
def plan_question(question: str):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    Convert the question into a SHORT search query (5-8 words max).
    No explanation. No quotes. Only keywords.

    Question: {question}
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    intent = response.content.strip()
    intent = intent.replace('"', '').replace("'", "")
    return intent


# -----------------------------
# Security Layer
# -----------------------------
def is_blocked_query(query: str):
    blocked_words = ["password", "secret", "key"]
    return any(word in query.lower() for word in blocked_words)


def is_sensitive_output(text: str):
    sensitive_words = ["password", "deeplearn2026"]
    return any(word in text.lower() for word in sensitive_words)


# -----------------------------
# Main RAG Pipeline
# -----------------------------
def query_rag(question: str):

    def sanitize_question(question: str):
        blocked_words = ["password", "secret", "key"]
        clean_question = question

        for word in blocked_words:
            clean_question = clean_question.replace(word, "")

        return clean_question.strip()

    # Get intent
    clean_question = sanitize_question(question)
    intent = plan_question(clean_question if clean_question else question)

    # Use API embeddings
    embeddings = get_embeddings()

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    docs = db.similarity_search(intent, k=3)

    context = "\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.page_content[:100] for doc in docs]))

    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    You are an intelligent AI assistant.

    Rules:
    - Answer ONLY from the context.
    - If answer is not found, say "Not available in the document."
    - Do NOT guess.

    Also classify confidence as HIGH, MEDIUM, or LOW.

    Context:
    {context}

    Question:
    {question}

    Return format:
    Answer: ...
    Confidence: ...
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    output = response.content

    answer = output
    confidence = "MEDIUM"

    if "Confidence:" in output:
        parts = output.split("Confidence:")
        answer = parts[0].replace("Answer:", "").strip()
        confidence = parts[1].strip()

    # Block sensitive output
    if is_sensitive_output(answer):
        return {
            "answer": "This information cannot be disclosed.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "Sensitive content blocked"
        }

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "intent_used": intent
    }

# Alternative way
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage
# import os


# # -----------------------------
# # Planner (Agent Layer)
# # -----------------------------
# def plan_question(question: str):
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant",
#         api_key=os.getenv("GROQ_API_KEY")
#     )

#     prompt = f"""
#     Convert the question into a SHORT search query (5-8 words max).
#     No explanation. No quotes. Only keywords.

#     Question: {question}
#     """

#     response = llm.invoke([
#         HumanMessage(content=prompt)
#     ])

#     intent = response.content.strip()
#     intent = intent.replace('"', '').replace("'", "")
#     return intent


# # -----------------------------
# # Security Layer
# # -----------------------------
# def is_blocked_query(query: str):
#     blocked_words = ["password", "secret", "key"]
#     return any(word in query.lower() for word in blocked_words)


# def is_sensitive_output(text: str):
#     sensitive_words = ["password", "deeplearn2026"]
#     return any(word in text.lower() for word in sensitive_words)


# # -----------------------------
# # Main RAG Pipeline
# # -----------------------------
# def query_rag(question: str):

#     # Blocks unsafe queries BEFORE anything
#     def sanitize_question(question: str):
#         blocked_words = ["password", "secret", "key"]

#         clean_question = question

#         for word in blocked_words:
#             clean_question = clean_question.replace(word, "")

#         return clean_question.strip()

#     # Get refined intent
#     clean_question = sanitize_question(question)
#     intent = plan_question(clean_question if clean_question else question)

#     # Embeddings + DB
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     db = Chroma(
#         persist_directory="db",
#         embedding_function=embeddings
#     )

#     docs = db.similarity_search(intent, k=3)

#     context = "\n".join([doc.page_content for doc in docs])

#     # Sources (clean duplicates)
#     sources = list(set([doc.page_content[:100] for doc in docs]))

#     # LLM
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant",
#         api_key=os.getenv("GROQ_API_KEY")
#     )

#     prompt = f"""
#     You are an intelligent AI assistant.

#     Rules:
#     - Answer ONLY from the context.
#     - If answer is not found, say "Not available in the document."
#     - Do NOT guess.

#     Also classify confidence as HIGH, MEDIUM, or LOW.

#     Context:
#     {context}

#     Question:
#     {question}

#     Return format:
#     Answer: ...
#     Confidence: ...
#     """

#     response = llm.invoke([
#         HumanMessage(content=prompt)
#     ])

#     output = response.content

#     # Parse response
#     answer = output
#     confidence = "MEDIUM"

#     if "Confidence:" in output:
#         parts = output.split("Confidence:")
#         answer = parts[0].replace("Answer:", "").strip()
#         confidence = parts[1].strip()

#     # Block sensitive OUTPUT
#     if is_sensitive_output(answer):
#         return {
#             "answer": "This information cannot be disclosed.",
#             "sources": [],
#             "confidence": "LOW",
#             "intent_used": "Sensitive content blocked"
#         }

#     # Final response
#     return {
#         "answer": answer,
#         "sources": sources,
#         "confidence": confidence,
#         "intent_used": intent
#     }