import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.embeddings import get_embeddings

def ingest_text(text: str):
    # Better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)
    embeddings = get_embeddings()

    # FIX: Create a unique collection name for EVERY upload 
    # This prevents retrieving old/stale data from previous sessions
    unique_name = f"astra_col_{uuid.uuid4().hex}"

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=unique_name
    )

    return db, len(chunks)