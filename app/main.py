from fastapi import FastAPI
from routers import query
import logging 


# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,  # INFO seviyesindeki logları göster
    format="%(levelname)s:\\t %(message)s",  # Log formatını belirleyelim
)

logger = logging.getLogger(__name__)

# Artık logger'ı bu şekilde kullanabilirsiniz
logger.info("This is an info message from llm container.")


app = FastAPI(title="LLM FastAPI", version="1.0.0")

# Routers'ı dahil et
app.include_router(query.router, prefix="/chat", tags=["Chat"])

@app.get("/")
def root():
    return {"message": "Welcome to the LLM FastAPI application!"}