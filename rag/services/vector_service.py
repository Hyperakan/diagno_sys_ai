from uuid import uuid4
from models.models import QueryRequest
from utils.model_utils import get_embedding_model
from io import BytesIO
from unstructured.partition.auto import partition   
from weaviate.classes.data import DataObject
from utils.vector_db_utils import get_client
import os
import logging

def search_documents(query_obj: QueryRequest):
    embedding_model = get_embedding_model()
    query_vector = embedding_model.encode(query_obj.query).tolist()
    client = get_client()
    documents = client.collections.get(query_obj.collection_name)
    
    relevant_chunks = documents.query.hybrid(
        query=query_obj.query,
        vector=query_vector,
        limit=query_obj.top_k,
        alpha=0.5
    )
    
    return relevant_chunks
    

def index_document(content: str, collection_name: str):
    model = get_embedding_model()
    client = get_client()

    chunked_texts = chunk_documents(content)
    embeddings = [model.embed(text) for text in chunked_texts]
    collection = client.collection.get(collection_name)
    data_obj = list()
    for i, d in enumerate(chunked_texts):
        data_obj.append(DataObject(
            properties={"text": d},
            vector=embeddings[i]))
        
    uuids = collection.data.insert(data_obj)
    logging.info(f"Succesfully indexed {len(uuids)} chunks.")

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


def extract_content(filestream: BytesIO):
    elements = partition(file=filestream)
    return "\n\n".join([str(el) for el in elements])