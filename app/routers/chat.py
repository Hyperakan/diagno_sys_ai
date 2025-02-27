from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from services.llm import async_llm_stream_response
from services.retrieval import process_query
from models.models import ChatRequest
import logging
import asyncio

router = APIRouter(prefix="/chat")

@router.post("/answer")
async def answer(chat_request: ChatRequest):
    """
    Gelen soruya LLM üzerinden yanıt oluşturur ve StreamingResponse ile yanıtı token bazında (boşluk karakterleri de dahil)
    gönderir. Son tokenin sonuna ekstra newline eklenir.
    Bu örnekte, llm.stream metodu kullanılarak asenkron streaming uygulanmaktadır.
    """
    async def stream_response():
        try:
            yield f"Soru: {chat_request.question}\n"
            if chat_request.search:
                chunks = await process_query(query=chat_request.question)
                # Asenkron generator üzerinden token'ları tek tek yield ediyoruz.
                async for token in async_llm_stream_response(
                    chat_request.question,
                    chunks,
                    chat_request.model_name,
                    chat_request.temperature
                ):
                    # İsteğe bağlı: token'ı re.split ile işleyip boşluklar vs. koruyabilirsiniz.
                    yield token
                    await asyncio.sleep(0.05)  # Akışı yumuşatmak için isteğe bağlı gecikme
                # Son tokenin sonuna newline ekle
                yield "\n"
            else:
                yield "Arama yapılmadı.\n"
        except Exception as e:
            logging.error(f"Streaming yanıt hatası: {e}")
            yield f"Error: {str(e)}\n"

    return StreamingResponse(stream_response(), media_type="text/plain")
