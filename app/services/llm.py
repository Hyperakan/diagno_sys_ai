from langchain_ollama import OllamaLLM
import os

def generate_response(question: str, chunks) -> str:
    ollama_url = os.getenv("OLAMA_URL")
    llm = OllamaLLM(model='mistral',  base_url=ollama_url, temperature=0, )
    
    prompt = f"context: {chunks} question: {question} answer:"
    response = llm.invoke(prompt)
    return response
