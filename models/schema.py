from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class ChatRequest(BaseModel):
    user_input: str

class RAGRequest(BaseModel):
    query: str
    agent_type: str = "simple"
    temp_document: Optional[str] = None
    format_prompt: Optional[str] = None

class Settings(BaseModel):
    default_llm: str
    default_embedding: str
    default_agent: str
    chunk_size: int
    chunk_overlap: int

class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    temperature: Optional[float] = None

class EmbedConfig(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    url: Optional[str] = None
    timeout: Optional[int] = None

class AddLLMRequest(BaseModel):
    name: str
    config: LLMConfig

class AddEmbedRequest(BaseModel):
    name: str
    config: EmbedConfig

class TestModelRequest(BaseModel):
    name: str

class FileInfo(BaseModel):
    filename: str
    size: Optional[int] = None
    type: Optional[str] = None
    word_count: Optional[int] = None
    upload_timestamp: Optional[str] = None

class UploadResponse(BaseModel):
    message: List[str]
    successful_uploads: int
    total_files: int

class ChatResponse(BaseModel):
    reply: str

class RAGResponse(BaseModel):
    result: str
    agent_type: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class MessageResponse(BaseModel):
    message: str

class TestResponse(BaseModel):
    success: bool
    message: str

class LLMListResponse(BaseModel):
    llms: List[str]
    selected: str

class EmbedListResponse(BaseModel):
    embeddings: List[str]
    selected: str

class AgentListResponse(BaseModel):
    agents: List[str]