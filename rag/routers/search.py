from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile
from models.models import QueryRequest
from services.vector_service import search_documents
from services.vector_service import extract_content
from services.vector_service import index_document
from io import BytesIO
import logging

router = APIRouter(prefix="/rag")

@router.post("/search")
async def search(request: QueryRequest):
    """
    Weaviate ve SentenceTransformer ile sorgu işlemi.
    """
    try:
        results = search_documents(request = request)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/index-document")
async def index_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Dosyayı BytesIO ile bir byte stream'e dönüştür
        file_stream = BytesIO(contents)

        # Dosyanın içeriğini unstructured ile çıkar
        extracted_data = extract_content(file_stream)
        logging.info(f"{extracted_data}")
        index_document(content=extracted_data, collection_name="med_documents")
            
        return {"filename": file.filename, "message": "File indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
