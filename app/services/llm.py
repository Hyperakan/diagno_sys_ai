from utils.ollama_utils import OllamaClientFactory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import queue
import threading
import asyncio
from typing import List
from models.models import Message
import logging
from fastapi import HTTPException
from fastapi.responses import Response
import json

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

        # Build full prompt, including system, context, and conversation history
        system_prompt = (
            "Sen alanında tecrübeli bir doktorsun. Aşağıda bir hasta ile ettiğin bir sohbet bulunuyor."
            "<doktor> ve <hasta> olarak iki kişinin konuşması ve <bağlam> arasında soruyla alakalı bağlam var."
            "Sen <doktor> kişisisin. <hasta> bir hasta."
            "Sohbette bir sonraki cevabını üret."
        )

        # create_prompt now handles parsing of messages into tagged conversation
        prompt_text = build_prompt(messages, section_str, system_prompt)
        prompt_message = HumanMessage(content=prompt_text)
        logging.info(f"Prompt: {prompt_message.content}")
        for token in llm.stream([prompt_message]):
            if "<hasta>:" in token.content:
                break
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
    elif message.sender == "bot":
        return AIMessage(content=message.content)

def create_section(chunks):
    section = ""
    for chunk in chunks['results']:
        section += f"{chunk['context']}\n\n"
    return section

def build_prompt(messages: List[Message], section: str, system_prompt: str) -> str:
    """
    Sistemi, bağlamı ve sohbet geçmişini alıp tek bir prompt stringine çevirir.
    Mesajları <hasta> ve <doktor> tagları arasında çifttagi ile işaretler.
    """
    # Sistem bloğu
    prompt = f"\n{system_prompt}\n\n"
    # Bağlam bloğu
    prompt += f"<bağlam>\n{section.strip()}\n<bağlam>\n\n"
    # Sohbet geçmişi
    for msg in sorted(messages, key=lambda m: m.timestamp):
        content = msg.content.strip()
        if msg.sender == "user":
            prompt += f"<hasta>: {content}\n"
        else:
            prompt += f"<doktor>: {content}\n"
    # Doktorun bir sonraki cevabı için açılış tagı
    prompt += "<doktor>: "
    return prompt

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
        "Bir konuşma verildiğinde ona kısa ve açıklayıcı bir başlık öneren yardımcı bir asistansınız. "
        "Yalnızca başlıkla yanıtlayın, başka metin eklemeyin. 5 kelimeden fazla olmamalı. "
    )
    prompt = [SystemMessage(content=system_prompt)]
    
    for msg in sorted(messages, key=lambda m: m.timestamp):
        prompt.append(message_to_langchain_message(msg))

    # 3) Başlık isteğini ekle
    prompt.append(HumanMessage(content="Please provide a short chat title:"))

    try:
        # llm.generate çağrısını thread pool'da çalıştır
        loop = asyncio.get_running_loop()
        llm_result = await loop.run_in_executor(
            None,
            lambda: llm.invoke(prompt, stop=["\n"])
        )
        
        title = llm_result.content.strip()
        logging.info(f"Generated chat title: {title}")
        return Response(
            content=json.dumps({"name": title}, ensure_ascii=False).encode(encoding="utf-8"),
            media_type="application/json",
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    except Exception as e:
        logging.error(f"Error generating chat name: {e.with_traceback()}")
        raise HTTPException(status_code=500, detail="Error generating chat name")
    
def generate_analyze_response(prompt: str):
    analyzer_ollama_client = OllamaClientFactory.get_client(role="analyzer")
    try:
        response = analyzer_ollama_client.generate(
            messages=[[HumanMessage(content=prompt)]]
        )
        return response.generations[0][0].text.strip()
    except Exception as e:
        logging.error(f"Error generating analysis response: {e.with_traceback()}")
        raise HTTPException(status_code=500, detail="Error generating analysis response")
