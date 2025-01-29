from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest
from models.models import QueryRequest
from utils.model_utils import get_embedding_model
import os

def search_vectors(query: QueryRequest):
    """
    Qdrant üzerinde en yakın vektörleri arar.
    """
    # Sorguyu encode et
    embedding_model = get_embedding_model()
    query_vector = embedding_model.encode(query.query).tolist()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_client = QdrantClient(url=qdrant_url)
    # Qdrant'da arama yap
    search_result = qdrant_client.search(
        collection_name="documents",
        search_request=SearchRequest(
            vector=query_vector,
            limit=query.top_k
        )
    )

    return [hit.payload for hit in search_result]
