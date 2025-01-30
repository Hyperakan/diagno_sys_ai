from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile
from models.models import QueryRequest
from services.vector_service import search_documents
from services.vector_service import extract_content
from services.vector_service import index_document

router = APIRouter(prefix="/rag")

@router.post("/search")
async def search(request: QueryRequest):
    """
    Qdrant ve SentenceTransformer ile sorgu işlemi.
    """
    try:
        results = search_documents(request = request)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/index-document")
async def index_document(file: UploadFile = File(...)):
    try:
        with open(f"uploaded_{file.filename}", "wb") as f:
            content = extract_content(f)
            index_document(content=content, collection_name="med-documents")
            
        return {"filename": file.filename, "message": "File indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
