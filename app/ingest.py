import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.embeddings import get_embeddings
import chromadb # Import the base client

def ingest_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    embeddings = get_embeddings()

    # FORCE IN-MEMORY ONLY (The "Nuclear" Option for Stale Data)
    # This creates a totally fresh client every single time.
    ephemeral_client = chromadb.EphemeralClient()
    
    unique_name = f"astra_{uuid.uuid4().hex}"

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=ephemeral_client, # Pass the fresh client here
        collection_name=unique_name
    )

    return db, len(chunks)