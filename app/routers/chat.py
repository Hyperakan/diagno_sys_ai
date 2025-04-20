from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.llm import async_llm_stream_response
from services.llm import name_chat
from services.retrieval import process_query
from models.models import ChatData
import logging
import asyncio

router = APIRouter(prefix="/chat")

@router.post("/answer")
async def answer(chat_data: ChatData):
    """
    Gelen soruya LLM üzerinden yanıt oluşturur ve StreamingResponse ile yanıtı token bazında (boşluk karakterleri de dahil)
    gönderir. Son tokenin sonuna ekstra newline eklenir.
    Bu örnekte, llm.stream metodu kullanılarak asenkron streaming uygulanmaktadır.
    """
    async def stream_response():
        try:
            chunks = await process_query(query=max(chat_data.messages, key=lambda msg: msg.timestamp).content)

            async for token in async_llm_stream_response(
                chat_data.messages,
                chunks,
            ):
                yield token
                await asyncio.sleep(0.05)
            
            yield "\n"
            
        except Exception as e:
            logging.error(f"Streaming yanıt hatası: {e}")
            yield f"Error: {str(e)}\n"

    return StreamingResponse(stream_response(), media_type="text/plain")

@router.post("/get-name")
async def get_name(chat_data: ChatData):
    """
    Sohbete, LLM'in mesajları analiz ederek oluşturacağı kısa bir başlık döner.
    Eğer chatInfo.name zaten "Yeni Sohbet" dışındaysa, mevcut adı geri döner.
    """
    # Eğer kullanıcı zaten özelleştirilmiş bir isim vermişse, onu koru.
    if chat_data.chatInfo.name != "Yeni Sohbet":
        return {"name": chat_data.chatInfo.name}
    else:
        return await name_chat(chat_data.messages)                    