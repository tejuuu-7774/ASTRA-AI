from langchain_text_splitters import CharacterTextSplitter

def ingest_text(text: str):

    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    return chunks