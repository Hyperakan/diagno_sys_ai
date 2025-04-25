from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import Response
from models.models import ProspectusRequest
from services.llm import generate_analyze_response
import httpx
import requests
import re

import logging

router = APIRouter(prefix="/drug")

@router.post("/interaction")
async def analyze_prospectuses(request: ProspectusRequest):
    """
    Read plain-text prospectus files from disk and analyze potential interactions and side effects.
    Handles empty file content scenarios.
    """
    drug_names = []
    # merge request.current_prospectuses and request.new_prospectuses in drug_names
    for drug in request.drugs:
        if isinstance(drug, str):
            drug_names.append(drug)
        else:
            raise HTTPException(status_code=400, detail="Invalid prospectus format")
        
    url = "https://kt-finder-676470519300.europe-west1.run.app"
    json_payload = {
        "kullanilan_ilaclar":  drug_names,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json_payload, timeout=20)
            
            if response.status_code != 200:
                logging.info(f"Response status code: {response.status_code}, response content: {response.content}")
            
            res = response.json()
            logging.info(f"{res}")
            prospectuses = []
            for key in res.keys():
                prospectuses.append(f"{key}: {res[key]['content']}\n")
                
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")

    # Build prompt using file contents
    count = len(prospectuses)
    
    # 2. Her bir prospektüsü numaralandır ve bölümler halinde hazırla
    sections = "\n\n".join(
        f"### Prospektüs {i+1}\n{extract_sections(text=text)}"
        for i, text in enumerate(prospectuses)
    )

    # 3. Prompt’u oluştur
    prompt = (
        f"Ilaç prospektüsleri ({count} adet):\n\n"
        f"{sections}\n\n"
        "Bu ilaç prospektüslerini ayrı ayrı değerlendirerek\n"
        "Aşağıdaki markdown başlıklarını doldur:\n\n"
        "## 📋 Kullanılan İlaçlar\n\n"
        "## 🔄 İlaçlar Arası Olası Etkileşimler\n\n"
        "## ⚠️ Bu İlaçların Birbiri İle Birlikte Kullanımı\n"
    )
    logging.info(f"{prompt}")
    # Generate analysis
    try:
        analysis = generate_analyze_response(prompt)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")
    logging.info(f"{analysis}")
    return Response(
            content=analysis,
            media_type="text/markdown",
            headers={"Content-Type": "text/markdown; charset=utf-8"}
    )
    
def extract_sections(text):
    """
    text: tüm metin
    döndürür: her bir bölümün içeriğini bir sözlükte.
    """
    # Bölüm isimleri ve onlar arasındaki delimiters (başlangıç, bitiş)
    patterns = {
        "Etkin madde": (
            r"Etkin madde:", 
            r"Bu ilacı kullanmaya başlamadan önce bu KULLANMA TALİMATINI dikkatlice\s*okuyunuz, çünkü sizin için önemli bilgiler içermektedir\."
        ),
        "Diğer ilaçlarla birlikte kullanımı": (
            r"Diğer ilaçlar ile birlikte kullanımı", 
            r"nasıl kullanılır\?"
        ),
    }

    result = {}
    for name, (start, end) in patterns.items():
        # Her pattern için başlık ile bitiş arasındaki metni yakalayalım
        regex = re.compile(
            rf"{start}\s*(?P<body>.*?)(?={end})",
            flags=re.DOTALL | re.IGNORECASE
        )
        m = regex.search(text)
        if m:
            result[name] = m.group("body").strip()
        else:
            result[name] = None

    return result
