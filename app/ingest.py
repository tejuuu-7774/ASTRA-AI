import shutil
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.embeddings import get_embeddings


def ingest_text(text: str):

    # Clear old DB
    if os.path.exists("db"):
        shutil.rmtree("db")

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    # API embeddings (lightweight)
    embeddings = get_embeddings()

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    return len(chunks)