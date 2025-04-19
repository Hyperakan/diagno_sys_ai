from utils.ollama_utils import get_ollama_client
from langchain_core.messages import HumanMessage, AIMessage
import os
import queue
import threading
import asyncio
from typing import List
from models.models import Message
import logging

def stream_response_with_context_sync(messages: List[Message], chunks, output_queue: queue.Queue):
    """
    Ollama container'ına prompt göndermek için, gelen chunks'tan section (bağlam)
    oluşturur, kullanıcı mesajını prompt ile birleştirir ve llm.stream ile token'ları 
    output_queue'ya aktarır.
    """
    ollama_url = os.getenv("OLLAMA_URL")
    if not ollama_url:
        output_queue.put(ValueError("OLLAMA_URL environment variable is not set."))
        return

    llm = get_ollama_client()

    try:
        # chunks'tan section (bağlam) oluştur.
        section_str = create_section(chunks)
        section_msg = HumanMessage(content=section_str)

        # Mesajlar arasında son kullanıcı mesajını bul.
        last_user_message = max(messages, key=lambda msg: msg.timestamp)

        if not last_user_message:
            output_queue.put(ValueError("No user message found."))
            return
        user_msg = message_to_langchain_message(last_user_message)

        # Kullanıcı mesajı ve section'ı kullanarak prompt oluştur.
        system_prompt = """You are a healthcare assistant and you are answering a patient's question.
        Do not forget to be professional and ethical in your answers.
        Do not use citations or references in your answers.
        Do not use technical terms that the patient cannot understand.
        Do not ask any questions based on the context you provided.
        Always suggest the patient to consult a healthcare professional for a proper diagnosis and treatment after answering the question.
        If the question between two <q>'s does not require any knowledge to answer, ignore the provided content between two <ctx>'s."""
        
        prompt = build_prompt(user_msg, section_msg, system_prompt)
        prompt_message = HumanMessage(content=prompt)
        for token in llm.stream([prompt_message]):
            output_queue.put(token.content)
    except Exception as e:
        output_queue.put(RuntimeError(f"LLM streaming çağrısında hata oluştu: {e}"))
    finally:
        output_queue.put(None)

async def async_llm_stream_response(messages: List[Message], chunks):
    """
    Senkron stream'i arka planda çalışan bir thread ve queue ile asenkron generator'a çevirir.
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

def create_section(chunks):
    """
    LLM'den gelen chunks verisini kullanarak section (bağlam) oluşturur.
    Beklenen yapı: chunks içerisinde "query" ve "results" (her biri {id, context, score}) bulunur.
    """
    section = ""
    for chunk in chunks['results']:
        section += f"{chunk['context']}\n\n"
    return section

def build_prompt(message: HumanMessage, section: HumanMessage, system_prompt: str):
    """
    Oluşturulan section, sistem prompt ve kullanıcı mesajını kullanarak LLM için prompt oluşturur.
    """
    return (
        f"<sys>\n{system_prompt}\n<sys>\n\n"
        f"<ctx>\n{section.content}<ctx>\n\n"
        f"<q>\n{message.content}\n<q>\n\n"
        "Answer:"
    )

