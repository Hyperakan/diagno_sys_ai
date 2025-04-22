import requests
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path

# Ollama server configuration
HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"
our_path = Path(__file__).parent.parent.resolve()

# Directory containing prospectus files
PROSPECTUS_DIR = our_path / "prospectuses"


# Directory containing prospectus files
PROSPECTUS_DIR = our_path / "prospectuses"

def pull_model():
    """Pull the model from Ollama server if not already present."""
    print(f"Pulling model '{MODEL_NAME}' if needed...")
    try:
        response = requests.post(
            f"{HOST}/api/pull", json={"model": MODEL_NAME}, stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(line)
    except requests.RequestException as e:
        print(f"[Warning] Model pull failed: {e}")


def generate_response(prompt: str) -> str:
    """
    Send a prompt to the Ollama generation endpoint and return the response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
        #"options": {"num_ctx": 100000}
    }
    resp = requests.post(f"{HOST}/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify Ollama server and pull model
    try:
        resp = requests.get(f"{HOST}/api/version")
        resp.raise_for_status()
        print("Ollama server version:", resp.json())
    except requests.RequestException as e:
        print(f"[Warning] Version check failed: {e}")

    # Pull model
    pull_model()
    
    yield
    
    # Cleanup (if needed)
    pass

app = FastAPI(lifespan=lifespan)

class ProspectusRequest(BaseModel):
    current_prospectuses: List[str] = Field(..., alias="current prospectuses")
    new_prospectus: str = Field(..., alias="new prospectus")

    class Config:
        allow_population_by_alias = True
        populate_by_name = True

@app.post("/analyze")
async def analyze_prospectuses(request: ProspectusRequest):
    """
    Read plain-text prospectus files from disk and analyze potential interactions and side effects.
    Handles empty file content scenarios.
    """
    # Read existing prospectus contents
    current_texts: List[str] = []
    for filename in request.current_prospectuses:
        file_path = PROSPECTUS_DIR / filename
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
        analysis = generate_response(prompt)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")

    return {"prompt":prompt, "analysis": analysis}
