from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile
from models.models import QueryRequest
from services.vector_service import search_documents
from services.vector_service import embed_and_index_documents
import os
import logging
router = APIRouter(prefix="/rag")

@router.post("/search")
async def search(request: QueryRequest):
    """
    Weaviate ve SentenceTransformer ile sorgu işlemi.
    """
    try:
        results = search_documents(query_obj=request)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/index-document")
async def index_document(file: UploadFile = File(...)):
    try:
        byte_data = await file.read()
        content = byte_data.decode("utf-8")
        embed_and_index_documents(content=content, collection_name=os.getenv("COLLECTION_NAME"))
            
        return {"filename": file.filename, "message": "File indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()
