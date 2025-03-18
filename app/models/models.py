from datetime import datetime 
from pydantic import BaseModel
from typing import List, Optional

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
    