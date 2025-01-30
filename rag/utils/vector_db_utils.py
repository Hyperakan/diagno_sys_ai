from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os
import logging

def create_client():
    global _qdrant_client
    qdrant_url = os.getenv("QDRANT_URL")
    _qdrant_client = QdrantClient(url=qdrant_url)
    return _qdrant_client
        
def get_client():
    if _qdrant_client is None:
        raise RuntimeError("Qdrant client is not loaded!")
    return _qdrant_client
    
def create_collection(collection_name: str):
    client = get_client()
    try:
        client.collection_exists(collection_name)
    except Exception as e:
        logging.info(f"Collection {collection_name} already exists.")
        pass
    client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=1024,
                                                                                         distance=Distance.COSINE))