from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.ollama_utils import OllamaClientFactory
from routers import chat
from routers import prospectus
import logging 
import os 


logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s:\t  %(message)s",  
)

logger = logging.getLogger(__name__)

logger.info("This is an info message from llm container.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI uygulaması başlatılırken modeli yükler ve weaviate collection olusturur, uygulama sonlandığında ise boşaltır.
    """
    try:
        logging.info("Starting lifespan context - creating OLLAMA client.")
        chat_model_name = os.getenv("CHAT_MODEL_NAME")
        namer_model_name = os.getenv("NAMER_MODEL_NAME")
        analyzer_model_name = os.getenv("ANALYZER_MODEL_NAME")
        ollama_url = os.getenv("OLLAMA_URL")
        temperature = float(os.getenv("TEMPERATURE"))
        if chat_model_name is None or namer_model_name is None or ollama_url is None or temperature is None:
            raise RuntimeError("CHAT_MODEL_NAME, NAMER_MODEL_NAME, OLLAMA_URL, and TEMPERATURE environment variables must be set to use OLLAMA.")
        else:
            OllamaClientFactory.create_client(
                role="chat",
                model_name=chat_model_name,
                base_url=ollama_url,
                temperature=temperature
            )
            OllamaClientFactory.create_client(
                role="namer",
                model_name=namer_model_name,
                base_url=ollama_url,
                temperature=temperature
            )
            OllamaClientFactory.create_client(
                role="analyzer",
                model_name=namer_model_name,
                base_url=ollama_url,
                temperature=temperature
            )
        yield 
    except Exception as e:
        logging.error(f"Error during lifespan: {e}")
        raise e
    finally:
        logging.info("Ending lifespan context - deleting OLLAMA client.")
        OllamaClientFactory.delete_client(role="chat")
        OllamaClientFactory.delete_client(role="namer")
        OllamaClientFactory.delete_client(role="analyzer")
        

app = FastAPI(title="LLM FastAPI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

app.include_router(chat.router, prefix="", tags=["Chat"])
app.include_router(prospectus.router, prefix="", tags=["Prospectus"])

@app.get("/")
def root():
    return {"message": "Welcome to the LLM FastAPI application!"}