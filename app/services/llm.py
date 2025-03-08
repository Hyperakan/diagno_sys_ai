from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
import queue
import threading
import asyncio
from typing import List
from models.models import Message
import logging

def stream_response_with_context_sync(messages: List[Message], chunks, output_queue: queue.Queue, model_name: str = "llama3.2:1b", temperature: float = 0.4):
    """
    Senkron olarak Ollama container'ına .stream metodu ile prompt gönderir,
    gelen token’ları output_queue'ya koyar.
    """
    ollama_url = os.getenv("OLLAMA_URL")
    if not ollama_url:
        output_queue.put(ValueError("OLLAMA_URL environment variable is not set."))
        return

    llm = ChatOllama(model=model_name, base_url=ollama_url, temperature=temperature)
    """prompt = (
        f"Context:\n{chunks}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )"""
    try:
        # Burada llm.stream metodu, prompt'a karşılık gelen yanıtı token bazında (örneğin kelime veya token düzeyinde) üreten senkron bir generator döndürmelidir.
        context = chunks['results']['objects'][0]['properties']['context']
        
        for token in llm.stream(([message_to_langchain_message(message) for message in messages]+[(HumanMessage(content=context))])):
            output_queue.put(token.content)
    except Exception as e:
        output_queue.put(RuntimeError(f"LLM streaming çağrısında hata oluştu: {e}"))
    finally:
        # Streaming tamamlandığında None göndererek bitiş sinyali veriyoruz.
        output_queue.put(None)

async def async_llm_stream_response(messages: List[Message], chunks):
    """
    Synchronous stream'i, arka planda çalışan bir thread ve queue ile asenkron generator'a çevirir.
    """
    output_queue = queue.Queue()
    thread = threading.Thread(
        target=stream_response_with_context_sync,
        args=(messages, chunks, output_queue)
    )
    thread.start()

    loop = asyncio.get_running_loop()
    while True:
        token = await loop.run_in_executor(None, output_queue.get)
        if token is None:
            break
        if isinstance(token, Exception):
            raise token
        yield token
        
def message_to_langchain_message(message: Message):
    if message.sender == "user":
        return HumanMessage(content=message.content)
    else:
        return AIMessage(content=message.content)