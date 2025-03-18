from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.ollama_utils import create_ollama_client, delete_ollama_client
from routers import chat
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
        model_name = os.getenv("MODEL_NAME")
        ollama_url = os.getenv("OLLAMA_URL")
        temperature = float(os.getenv("TEMPERATURE"))
        if model_name is None or ollama_url is None or temperature is None:
            raise RuntimeError("MODEL_NAME, OLLAMA_URL, and TEMPERATURE environment variables must be set to use OLLAMA.")
        else:
            create_ollama_client(model_name=os.getenv("MODEL_NAME"), ollama_url=os.getenv("OLLAMA_URL"), temperature=float(os.getenv("TEMPERATURE")))
        yield 
    except Exception as e:
        logging.error(f"Error during lifespan: {e}")
        raise e
    finally:
        logging.info("Ending lifespan context - deleting OLLAMA client.")
        delete_ollama_client()
        

app = FastAPI(title="LLM FastAPI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

app.include_router(chat.router, prefix="", tags=["Chat"])

@app.get("/")
def root():
    return {"message": "Welcome to the LLM FastAPI application!"}