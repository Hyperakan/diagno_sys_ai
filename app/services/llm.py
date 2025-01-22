from langchain_ollama import OllamaLLM
import os

def generate_response(question: str) -> str:
    ollama_url = os.getenv("OLAMA_URL")
    llm = OllamaLLM(model='mistral',  base_url=ollama_url)
    
    prompt = f"question: {question} answer:"
    response = llm.invoke(prompt)
    return response
