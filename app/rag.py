from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


#  Planner (Agent Layer)
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

#  Main RAG Pipeline
def query_rag(question: str):
    #  Get refined intent
    intent = plan_question(question)

    #  Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #  Load DB
    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    #  Use intent instead of raw question
    docs = db.similarity_search(intent, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    #  Remove duplicate sources
    sources = list(set([doc.page_content[:100] for doc in docs]))

    #  LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
        You are an intelligent AI assistant.
        Answer clearly based ONLY on the context.
        Also classify confidence as HIGH, MEDIUM, or LOW.
        Context: {context}
        User Question: {question}
        Return format:
        Answer: ...
        Confidence: ...
        """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    output = response.content

    # Parse response
    answer = output
    confidence = "MEDIUM"

    if "Confidence:" in output:
        parts = output.split("Confidence:")
        answer = parts[0].replace("Answer:", "").strip()
        confidence = parts[1].strip()

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "intent_used": intent   
    }