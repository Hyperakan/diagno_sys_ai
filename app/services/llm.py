from utils.ollama_utils import OllamaClientFactory
from langchain_core.messages import HumanMessage, AIMessage
import os
import queue
import threading
import asyncio
from typing import List
from models.models import Message
import logging
from fastapi import HTTPException

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

    llm = OllamaClientFactory.get_client(role="chat")

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

async def name_chat(messages: List[Message]):
    """
    LLM'e sohbet başlığı oluşturması için bir istek gönderir.
    """
    # Ollama'ya bağlanmak için URL kontrolü
    if not os.getenv("OLLAMA_URL"):
        raise ValueError("OLLAMA_URL environment variable is not set.")

    llm = OllamaClientFactory.get_client(role="namer")

    # 1) Sistem prompt'u
    system_prompt = (
        "You are a helpful assistant that, given a conversation, "
        "proposes a concise and descriptive title for it. "
        "Respond with only the title, no extra text."
    )
    messages = [HumanMessage(content=system_prompt)]
    
    for msg in sorted(messages, key=lambda m: m.timestamp):
        messages.append(message_to_langchain_message(msg))

    # 3) Başlık isteğini ekle
    messages.append(HumanMessage(content="Please provide a short chat title:"))

    try:
        # llm.generate çağrısını thread pool'da çalıştır
        loop = asyncio.get_running_loop()
        llm_result = await loop.run_in_executor(
            None,
            lambda: llm.generate(messages)
        )

        # LangChain LLMResult → generations[0][0].text
        title = llm_result.generations[0][0].text.strip()
        return {"name": title}

    except Exception as e:
        logging.error(f"Error generating chat name: {e}")
        raise HTTPException(status_code=500, detail="Error generating chat name")