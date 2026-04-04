from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_text(text: str):
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

# Breaks into pieces (chunks)
# Converts into embeddings (AI understanding)
# Stores in vector DB