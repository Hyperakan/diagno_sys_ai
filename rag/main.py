from fastapi import FastAPI
from routers import search
from contextlib import asynccontextmanager
from utils.model_utils import load_embedding_model, unload_embedding_model
from utils.vector_db_utils import create_client, create_collection
import logging 
import os

# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,  # INFO seviyesindeki logları göster
    format="%(levelname)s:\t  %(message)s",  # Log formatını belirleyelim
)

logger = logging.getLogger(__name__)

# Artık logger'ı bu şekilde kullanabilirsiniz
logger.info("This is an info message from rag container.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI uygulaması başlatılırken modeli yükler ve qdrant collection olusturur, uygulama sonlandığında ise boşaltır.
    """
    try:
        logging.info("Starting lifespan context - loading model.")
        load_embedding_model()  # Modeli yükle
        logging.info("Creating Qdrant client and collection.")
        create_client()
        create_collection(collection_name="med-documents")
        yield 
    except Exception as e:
        logging.error(f"Error during lifespan: {e}")
        raise e
    finally:
        logging.info("Ending lifespan context - unloading model.")
        unload_embedding_model()  # Uygulama sonlanınca modeli boşalt

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

# Router'ı dahil et
app.include_router(search.router, prefix="/rag", tags=["Rag"])

@app.get("/")
def root():
    return {"message": "Welcome to the RAG API!"}