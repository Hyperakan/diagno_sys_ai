from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    model_name: str
    temperature: float
    search: bool
    