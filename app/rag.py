from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os

def is_sensitive_output(text: str):
    sensitive_words = ["password", "secret", "key"]
    return any(word in text.lower() for word in sensitive_words)

def query_rag(question: str, vector_db):
    if vector_db is None:
        return {
            "answer": "Please upload a PDF first.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": "No DB"
        }

    # 🔹 STEP 1: Vector retrieval (Semantic search)
    docs = vector_db.similarity_search(question, k=8)

    # 🔹 STEP 2: Keyword Boost (SMART + SAFE)
    try:
        all_docs = vector_db._collection.get()["documents"]
    except:
        all_docs = []

    # Detect if question is about timings/schedules
    time_queries = ["time", "timing", "schedule", "monday", "friday", "hour", "when", "open"]
    is_time_request = any(k in question.lower() for k in time_queries)

    class Doc:
        def __init__(self, text):
            self.page_content = text

    keyword_docs = []

    if is_time_request:
        count = 0
        # Specific keywords to find the schedule chunk
        for t in all_docs:
            if any(k in t.lower() for k in ["monday", "friday", "timing", "schedule"]):
                keyword_docs.append(Doc(t))
                count += 1
                if count >= 5: # Limit to 5 chunks to avoid overwhelming the LLM
                    break

    # Merge Vector results + Keyword results and deduplicate
    combined_docs = []
    seen = set()

    for d in (docs + keyword_docs):
        if d.page_content not in seen:
            combined_docs.append(d)
            seen.add(d.page_content)

    if not combined_docs:
        return {
            "answer": "Not available in the document.",
            "sources": [],
            "confidence": "LOW",
            "intent_used": question
        }

    # 🔹 STEP 3: Build context safely
    MAX_CONTEXT_CHARS = 2500
    context = ""
    used_docs = []

    for d in combined_docs:
        if len(context) + len(d.page_content) > MAX_CONTEXT_CHARS:
            break
        context += d.page_content + "\n"
        used_docs.append(d)

    sources = list(set([d.page_content[:120] + "..." for d in used_docs]))

    # 🔹 STEP 4: LLM Generation
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
    Answer ONLY from the context provided. 
    If the answer is not in the context, say "Not available in the document."

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
        answer = f"Error: {str(e)}"

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
        "confidence": "MEDIUM" if sources else "LOW",
        "intent_used": question
    }