from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile
from models.models import QueryRequest
from models.models import IndexDocumentRequest
from services.vector_service import search_documents
from services.vector_service import index_document

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
async def index_document(index_document_request: IndexDocumentRequest):
    try:
        file = index_document_request.file
        collection_name = index_document_request.collection_name
        content = await file.read()
        index_document(content=content, collection_name=collection_name)
            
        return {"filename": file.filename, "message": "File indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
