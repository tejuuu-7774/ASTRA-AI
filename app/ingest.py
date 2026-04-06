from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.embeddings import get_embeddings


def ingest_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)

    embeddings = get_embeddings()

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return db, len(chunks)