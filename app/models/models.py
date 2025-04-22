from datetime import datetime 
from pydantic import BaseModel, Field
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
    
class ProspectusRequest(BaseModel):
    current_prospectuses: List[str] = Field(..., alias="current prospectuses")
    new_prospectus: str = Field(..., alias="new prospectus")

    class Config:
        allow_population_by_alias = True
        populate_by_name = True
    