from fastapi import APIRouter, HTTPException
from services.llm import generate_response_with_context
from services.retrieval import process_query
from models.models import ChatRequest
import logging
router = APIRouter(prefix="/chat")

@router.post("/answer")
async def answer(chat_request: ChatRequest):
    """
    Gelen soruya LLM üzerinden yanıt oluşturur.
    """
    try:
        response = "Yanit Olusturulamadi."
        if chat_request.search:
            chunks = await process_query(query=chat_request.question)
            response = generate_response_with_context(chat_request.question, 
                                        chunks, 
                                        chat_request.model_name, 
                                        chat_request.temperature)
        return {"question": chat_request.question, "response": response}
    except Exception as e:
        logging.info(f"{e}")
        raise HTTPException(status_code=500, detail=str(e))