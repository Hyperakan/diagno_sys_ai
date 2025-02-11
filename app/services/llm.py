from langchain_ollama import OllamaLLM
import os

def generate_response_with_context(question: str, chunks, model_name: str, temperature: float) -> str:
    ollama_url = os.getenv("OLLAMA_URL")
    llm = OllamaLLM(model=model_name,  base_url=ollama_url, temperature=temperature)
    
    prompt = f"context: {chunks}\n question: {question}\n answer:"
    response = llm.invoke(prompt)
    return response
