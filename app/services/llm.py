from langchain_ollama import OllamaLLM
import os
import queue
import threading
import asyncio
def stream_response_with_context_sync(question: str, chunks, model_name: str, temperature: float, output_queue: queue.Queue):
    """
    Senkron olarak Ollama container'ına .stream metodu ile prompt gönderir,
    gelen token’ları output_queue'ya koyar.
    """
    ollama_url = os.getenv("OLLAMA_URL")
    if not ollama_url:
        output_queue.put(ValueError("OLLAMA_URL environment variable is not set."))
        return

    llm = OllamaLLM(model=model_name, base_url=ollama_url, temperature=temperature)
    prompt = (
        f"Context:\n{chunks}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    try:
        # Burada llm.stream metodu, prompt'a karşılık gelen yanıtı token bazında (örneğin kelime veya token düzeyinde) üreten senkron bir generator döndürmelidir.
        for token in llm.stream(prompt):
            output_queue.put(token)
    except Exception as e:
        output_queue.put(RuntimeError(f"LLM streaming çağrısında hata oluştu: {e}"))
    finally:
        # Streaming tamamlandığında None göndererek bitiş sinyali veriyoruz.
        output_queue.put(None)

async def async_llm_stream_response(question: str, chunks, model_name: str, temperature: float):
    """
    Synchronous stream'i, arka planda çalışan bir thread ve queue ile asenkron generator'a çevirir.
    """
    output_queue = queue.Queue()
    thread = threading.Thread(
        target=stream_response_with_context_sync,
        args=(question, chunks, model_name, temperature, output_queue)
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