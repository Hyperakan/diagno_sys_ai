from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection_name: str = "med_documents"