from uuid import uuid4
from models.models import QueryRequest
from utils.model_utils import get_embedding_model
from weaviate.classes.data import DataObject
from utils.vector_db_utils import get_client
from typing import List
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
        alpha=query_obj.hybrid_alpha
    )
    
    return relevant_chunks
    

def embed_and_index_documents(content: str, collection_name: str):
        model = get_embedding_model()
        client = get_client()

        # Metni parçalara ayırıyoruz
        chunked_texts = chunk_text(content)

        embeddings = model.encode(chunked_texts, show_progress_bar=True)

        # Belirtilen koleksiyonu alıyoruz
        collection = client.collections.get(collection_name)

        # DataObject listesi oluşturuluyor
        data_objects = []
        for text, embedding in zip(chunked_texts, embeddings):
            properties={
                    "context": text
            }
            logging.info(type(properties))
            data_objects.append(DataObject(
                properties=properties,
                vector=embedding
            ))
        try:    
            uuids = collection.data.insert_many(data_objects)
        except Exception as e:
            logging.info(f"{e}")

def chunk_text(content: str, chunk_size: int = 512, overlap: int = 20) -> List[str]:
    model = get_embedding_model()

    return split_text_on_tokens(text=content, tokenizer=model.tokenizer, chunk_size=chunk_size, chunk_overlap=overlap)

def split_text_on_tokens(*, text: str, tokenizer, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Verilen metni, tokenizer kullanarak parçalara ayırır.
    Her parça, belirlenen token sayısına (chunk_size) göre oluşturulur,
    ve parçalar arasında belirli bir örtüşme (chunk_overlap) bulunur.
    Ayrıca, chunk_size değerinin modelin desteklediği maksimum sekans uzunluğunu aşmamasını sağlar.
    """
    try:
        splits: List[str] = []
        input_ids = tokenizer.encode(text)
        total_tokens = len(input_ids)
        
        # Tokenizer nesnesinde modelin desteklediği maksimum token uzunluğunu alıyoruz. 
        # Eğer bu özellik yoksa varsayılan değeri 8192 olarak alıyoruz.
        max_model_length = getattr(tokenizer, "model_max_length", 8192)
        
        # Eğer istenen chunk_size modelin maksimum uzunluğundan büyükse, uyarı verip chunk_size'ı küçültüyoruz.
        if chunk_size > max_model_length:
            logging.info(
                f"chunk_size ({chunk_size}) is greater than model's maximum sequence length ({max_model_length}). "
                f"Using {max_model_length} as chunk_size instead."
            )
            chunk_size = max_model_length

        start_idx = 0
        while start_idx < total_tokens:
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk_ids = input_ids[start_idx:end_idx]
            try:
                splits.append(tokenizer.decode(chunk_ids))
            except Exception as ex:
                logging.info(f"{ex}")
            if end_idx == total_tokens:
                break
            
            start_idx += chunk_size - chunk_overlap

        return splits

    except Exception as e:
        logging.exception("Error occurred while splitting text on tokens:")
        return []

