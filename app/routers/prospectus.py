from fastapi import APIRouter
from fastapi import HTTPException
from models.models import ProspectusRequest
from services.llm import generate_analyize_response
from typing import List
import requests
from pathlib import Path

our_path = Path(__file__).parent.parent.resolve()

# Directory containing prospectus files
PROSPECTUS_DIR = our_path / "prospectuses"

router = APIRouter(prefix="/prospectus")

@router.post("/analyze")
async def analyze_prospectuses(request: ProspectusRequest):
    """
    Read plain-text prospectus files from disk and analyze potential interactions and side effects.
    Handles empty file content scenarios.
    """
    # Read existing prospectus contents
    current_texts: List[str] = []
    for filename in request.current_prospectuses:
        file_path = PROSPECTUS_DIR / filename
        print(f"Reading file: {file_path}")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading {filename}: {e}")
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"Prospectus content is empty: {filename}")
        current_texts.append(text)

    # Read new prospectus
    new_path = PROSPECTUS_DIR / request.new_prospectus
    if not new_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.new_prospectus}")
    try:
        new_text = new_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {request.new_prospectus}: {e}")
    if not new_text.strip():
        raise HTTPException(status_code=400, detail=f"Prospectus content is empty: {request.new_prospectus}")

    # Build prompt using file contents
    prompt = (
        "Mevcut ilaç prospektüsleri:\n" + "\n---\n".join(current_texts) + "\n"
        "Yeni ilaç prospektusu:\n" + new_text + "\n"
        "Bu ilaçlar arasında olası etkileşimler, yan etkiler veya çakışmalar var mı? "
        "Detaylı bir analiz yap."
    )

    # Generate analysis
    try:
        analysis = generate_analyize_response(prompt)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")

    return {"prompt":prompt, "analysis": analysis}