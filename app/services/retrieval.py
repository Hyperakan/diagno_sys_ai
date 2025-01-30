import httpx
from fastapi import HTTPException
from models.models import Query

async def process_query(query: Query, top_k: int = 5):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/rag/search",  # RAG container endpoint
                json={"query": query.question, "top_k": top_k}
            )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")
