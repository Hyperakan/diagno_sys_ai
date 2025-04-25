from fastapi import APIRouter
from fastapi import HTTPException
from models.models import ProspectusRequest
from services.llm import generate_analyze_response
import httpx
import requests
from pathlib import Path

import logging

router = APIRouter(prefix="/prospectus")

@router.post("/analyze")
async def analyze_prospectuses(request: ProspectusRequest):
    """
    Read plain-text prospectus files from disk and analyze potential interactions and side effects.
    Handles empty file content scenarios.
    """
    drug_names = []
    # merge request.current_prospectuses and request.new_prospectuses in drug_names
    for prospectus in request.current_prospectuses:
        if isinstance(prospectus, str):
            drug_names.append(prospectus)
        else:
            raise HTTPException(status_code=400, detail="Invalid prospectus format")
    if isinstance(request.new_prospectus, str):
        drug_names.append(request.new_prospectus)
        
    url = "https://kt-finder-676470519300.europe-west1.run.app"
    json = {
        "kullanilan_ilaclar":  drug_names,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json, timeout=10)
            
            if response.status_code != 200:
                logging.info(f"Response status code: {response.status_code}, response content: {response.content}")
            
            res = response.json()
            prospectuses = []
            for key in res.keys():
                prospectuses.append(f"{key}: {res[key]['content']}\n")
                
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")

    # Build prompt using file contents
    prompt = (
        "Ilaç prospektüsleri:\n" + "\n---\n".join(prospectuses) + "\n"
        "Bu ilaçlar arasında olası etkileşimler, yan etkiler veya çakışmalar var mı? "
        "Detaylı bir analiz yap."
    )

    # Generate analysis
    try:
        analysis = generate_analyze_response(prompt)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")

    return {"prompt":prompt, "analysis": analysis}