from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat
import logging 


# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,  # INFO seviyesindeki logları göster
    format="%(levelname)s:\t  %(message)s",  # Log formatını belirleyelim
)

logger = logging.getLogger(__name__)

# Artık logger'ı bu şekilde kullanabilirsiniz
logger.info("This is an info message from llm container.")


app = FastAPI(title="LLM FastAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Routers'ı dahil et
app.include_router(chat.router, prefix="", tags=["Chat"])

@app.get("/")
def root():
    return {"message": "Welcome to the LLM FastAPI application!"}