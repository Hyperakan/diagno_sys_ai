from datetime import datetime 
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
class ChatRequest(BaseModel):
    question: str
    model_name: str
    temperature: float
    search: bool

class ChatInfo(BaseModel):
    id: Optional[str]  # ID might be empty or null
    name: str
    lastMessageTimestamp: datetime

class Message(BaseModel):
    id: str
    content: str
    timestamp: datetime
    sender: str
    
class ChatData(BaseModel):
    chatInfo: ChatInfo
    messages: List[Message]
    