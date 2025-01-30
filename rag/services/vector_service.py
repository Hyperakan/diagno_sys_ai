from uuid import uuid4
from models.models import QueryRequest
from utils.model_utils import get_embedding_model
from fastapi import File
from unstructured.partition.auto import partition   
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import os

def search_documents(query: QueryRequest):
    embedding_model = get_embedding_model()
    query_vector = embedding_model.encode(query.query).tolist()
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    hits = client.query_points(
        collection_name="med-documents",
        query=query_vector,
        limit=query.top_k,
    ).points
    
    return hits

def index_document(content: str, collection_name: str):
    model = get_embedding_model()
    client = QdrantClient(url=os.getenv("QDRANT_URL"))

    chunked_texts = chunk_documents(content)
    embeddings = [model.embed(text) for text in chunked_texts]

    points = [
    PointStruct(id=i, vector=embedding, payload={"text": text})
    for i, (embedding, text) in enumerate(zip(embeddings, chunked_texts))
    ]

    client.upsert(collection_name=collection_name, points=points)

    return f"Stored {len(points)} documents in {collection_name}"

def chunk_documents(content, chunk_size=512, overlap=20):
    model = get_embedding_model()
    tokens = model.tokenize(content)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        start += chunk_size - overlap  # Move forward with overlap
    
    chunked_texts = [model.untokenize(chunk) for chunk in chunks]
    
    return chunked_texts


def extract_content(file: File):
    elements = partition(filename=file.filename)
    return "\n\n".join([str(el) for el in elements])