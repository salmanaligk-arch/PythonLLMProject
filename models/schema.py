from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_input: str

class RAGRequest(BaseModel):
    query: str
    agent_type: str = "simple"
    temp_document: Optional[str] = None
    format_prompt: Optional[str] = None

#print(dir(BaseModel))