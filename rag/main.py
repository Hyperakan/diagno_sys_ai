from fastapi import FastAPI
from routers import search
from contextlib import asynccontextmanager
from utils.model_utils import load_embedding_model, unload_embedding_model
from utils.vector_db_utils import create_client, create_collection, close_client_conection
import logging 
import os

# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t  %(message)s", 
)

logger = logging.getLogger(__name__)

logger.info("This is an info message from rag container.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI uygulaması başlatılırken modeli yükler ve weaviate collection olusturur, uygulama sonlandığında ise boşaltır.
    """
    try:
        logging.info("Starting lifespan context - loading model.")
        load_embedding_model()  
        logging.info("Creating Weaviate client and collection.")
        create_client()
        create_collection(collection_name=os.getenv("COLLECTION_NAME"))
        yield 
    except Exception as e:
        logging.error(f"Error during lifespan: {e}")
        raise e
    finally:
        logging.info("Ending lifespan context - unloading model.")
        unload_embedding_model() 
        close_client_conection()

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

app.include_router(search.router, prefix="", tags=["Rag"])

@app.get("/")
def root():
    return {"message": "Welcome to the RAG API!"}