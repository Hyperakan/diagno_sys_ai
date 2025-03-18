from langchain_ollama import ChatOllama
import logging

_ollama_client = None

def create_ollama_client(model_name, ollama_url, temperature):
    global _ollama_client
    
    try:
        _ollama_client = ChatOllama(model=model_name, base_url=ollama_url, temperature=temperature)
        logging.info("OLLAMA client created.")
        return _ollama_client
    
    except Exception as e:
        logging.error(f"Error creating OLLAMA client: {e}")
        raise RuntimeError(f"Error creating OLLAMA client: {e}")
    
def get_ollama_client():
    if _ollama_client is None:
        raise RuntimeError("OLLAMA client is not created!")
    return _ollama_client

def delete_ollama_client():
    global _ollama_client
    _ollama_client = None