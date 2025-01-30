from pydantic import BaseModel

class Query(BaseModel):
    question: str
    
class User(BaseModel):
    username: str
    email: str
    full_name: str = None
    disabled: bool = None