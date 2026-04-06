import os
import uuid  # Add this import
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

    # FIX: Use a unique collection name so old PDF data is ignored
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{uuid.uuid4().hex}" 
    )

    return db, len(chunks)