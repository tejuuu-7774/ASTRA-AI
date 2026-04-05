from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model="sentence-transformers/all-MiniLM-L6-v2"
    )