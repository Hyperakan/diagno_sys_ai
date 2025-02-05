from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    model_name: str
    temperature: str
    search: bool
    