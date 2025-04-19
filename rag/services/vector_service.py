from uuid import uuid4
from models.models import QueryRequest
from utils.model_utils import get_embedding_model
from utils.model_utils import get_reranker_model
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.internal import QueryReturn
from utils.vector_db_utils import get_client
from typing import List
import torch
import logging

def search_documents(query_obj: QueryRequest):  
    model = get_embedding_model()
    query_vector = model.encode(query_obj.query).tolist()
    client = get_client()
    documents = client.collections.get(query_obj.collection_name)
    
    relevant_chunks = documents.query.hybrid(
        query=query_obj.query,
        vector=query_vector,
        limit=query_obj.top_k,
        alpha=query_obj.hybrid_alpha,
        return_metadata=MetadataQuery(score=True)
    )
    
    return relevant_chunks

def rerank_documents(query_obj: QueryRequest, context_and_scores: List[dict]):
    try: 
        model = get_reranker_model()
        model.model.eval()
        query = query_obj.query
        contexts = [result["context"] for result in context_and_scores]
        query_context_pairs = [[query, context] for context in contexts]
        
        with torch.no_grad():
            inputs = model.tokenizer(query_context_pairs, padding=True, truncation=True, return_tensors="pt")
            device = next(model.model.parameters()).device  # Modelin çalıştığı cihazı al
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Girdileri ilgili cihaza taşı
            scores = model.model(**inputs).logits.view(-1).tolist()
        
        # Her belgeye ilgili skoru ekle
        for i, result in enumerate(context_and_scores):
            result["score"] = scores[i]
        
        # Skoru -1'in altında olan belgeleri hariç tut
        filtered_results = [result for result in context_and_scores if result["score"] >= -1]
        
        # Kalan sonuçları skora göre azalan sırada yeniden sırala
        reranked_search_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)
        return reranked_search_results
    except Exception as e:
        logging.error(f"Error during reranking: {e}")
        
    
def construct_context_and_score_list(search_results: QueryReturn):
    try:
        context_and_scores = []
        for result in search_results.objects:
            context_and_scores.append({
                "id": result.uuid.int,
                "context": result.properties["context"],
                "score": result.metadata.score
            })
        return context_and_scores
    except Exception as e:
        logging.error(f"Error during constructing context and score dict: {e}")
       

def embed_and_index_documents(content: str, collection_name: str):
        model = get_embedding_model()
        client = get_client()

        chunked_texts = chunk_text(content)

        embeddings = model.encode(chunked_texts, show_progress_bar=True)

        collection = client.collections.get(collection_name)

        data_objects = []
        for text, embedding in zip(chunked_texts, embeddings):
            properties={
                    "context": text
            }
            data_objects.append(DataObject(
                properties=properties,
                vector=embedding
            ))
        try:    
            uuids = collection.data.insert_many(data_objects)
        except Exception as e:
            logging.info(f"Error while inserting data to collection: {e}")

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
        
        max_model_length = getattr(tokenizer, "model_max_length", 8192)
        
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
        logging.exception(f"Error occurred while splitting text on tokens: {e}")
        return []

