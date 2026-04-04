import shutil
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def ingest_text(text: str):

    # Clears old database (fresh start every time)
    if os.path.exists("db"):
        shutil.rmtree("db")

    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_texts(
        chunks,
        embeddings,
        persist_directory="db"
    )

    return len(chunks)

# deletes old DB
# creates fresh one
# stores ONLY current document