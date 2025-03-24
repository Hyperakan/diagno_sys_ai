from pydantic import BaseModel
from fastapi import File, UploadFile
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection_name: str = os.getenv("COLLECTION_NAME")
    hybrid_alpha: float = 0.5

class IndexDocumentRequest(BaseModel):
    file: UploadFile = File(...)
    collection_name: str = os.getenv("COLLECTION_NAME")
    
class RerankerModel():
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    use_fp16: bool = False
    
