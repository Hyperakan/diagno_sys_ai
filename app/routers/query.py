from fastapi import APIRouter, HTTPException
from services.llm import generate_response
from services.retrieval import process_query
from models.models import Query

router = APIRouter(prefix="/chat")

@router.post("/answer")
async def answer(query: Query):
    """
    Gelen soruya LLM üzerinden yanıt oluşturur.
    """
    try:
        chunks = process_query(query=query)
        response = generate_response(query.question)
        return {"question": query.question, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))