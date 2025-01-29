from fastapi import APIRouter, HTTPException
from models.models import QueryRequest
from services.vector_service import search_vectors

router = APIRouter(prefix="/rag")

@router.post("/search")
async def search_endpoint(request: QueryRequest):
    """
    Qdrant ve SentenceTransformer ile sorgu işlemi.
    """
    try:
        results = search_vectors(request = request)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
