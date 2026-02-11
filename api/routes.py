from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from models.schema import (
    ChatRequest, RAGRequest, Settings, LLMListResponse, EmbedListResponse, AgentListResponse, TestResponse,
    AddLLMRequest, AddEmbedRequest, TestModelRequest, FileInfo,
    UploadResponse, ChatResponse, RAGResponse, MessageResponse
)
from services.chatbot import call_ai
from services.rag_agents import agent_manager
from services.vector_store import vector_store
from services.logger import logger
from services.llm_manager import llm_manager
from services.embed_manager import embed_manager
from frontend.settings_helpers import save_general_settings, load_settings
from frontend.file_operations import get_files_list, delete_selected_file
import os
import json

app = FastAPI()

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    return {"reply": call_ai(request.user_input)}

@app.post("/rag/search", response_model=RAGResponse)
def rag_search(request: RAGRequest):
    try:
        # Use AgentManager for efficient agent creation
        agent = agent_manager.create_agent(request.agent_type, verbose=False)
        
        result = agent.process(
            query=request.query,
            temp_document=request.temp_document,
            format_prompt=request.format_prompt
        )
        return {"result": result, "agent_type": request.agent_type}
    except Exception as e:
        logger.error(f"RAG search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        results = []
        successful_uploads = 0
        
        for file in files:
            try:
                content = await file.read()
                filename = file.filename or "unknown_file"
                
                if not filename.endswith(('.pdf', '.txt', '.xlsx', '.docx')):
                    results.append(f"Unsupported file type: {filename}")
                    continue
                
                success = vector_store.add_file(content, filename)
                if success:
                    results.append(f"Successfully uploaded and indexed: {filename}")
                    successful_uploads += 1
                else:
                    results.append(f"Failed to process: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing file {getattr(file, 'filename', 'unknown')}: {str(e)}")
                results.append(f"Error processing {getattr(file, 'filename', 'unknown')}: {str(e)}")
        
        # Save index once for all successful uploads
        if successful_uploads > 0:
            vector_store.save_index()
            logger.info(f"Successfully processed {successful_uploads}/{len(files)} files")
        
        return UploadResponse(
            message=results,
            successful_uploads=successful_uploads,
            total_files=len(files)
        )
    
    except Exception as e:
        logger.error(f"Upload batch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Multi-Agent RAG System API"}

# Settings endpoints
@app.get("/settings", response_model=Settings)
def get_settings():
    try:
        llm, embed, agent, chunk_size, chunk_overlap, _ = load_settings()
        return Settings(
            default_llm=llm or llm_manager.get_selected(),
            default_embedding=embed or embed_manager.get_selected(),
            default_agent=agent or "simple",
            chunk_size=chunk_size or 1000,
            chunk_overlap=chunk_overlap or 100
        )
    except Exception as e:
        logger.error(f"Failed to get settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")

@app.post("/settings", response_model=MessageResponse)
def update_settings(settings: Settings):
    try:
        # Use the existing save function
        result = save_general_settings(
            settings.default_llm,
            settings.default_embedding,
            settings.default_agent,
            settings.chunk_size,
            settings.chunk_overlap
        )
        return {"message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

# LLM Management endpoints
@app.get("/llms", response_model=LLMListResponse)
def list_llms():
    try:
        return {"llms": llm_manager.get_names(), "selected": llm_manager.get_selected()}
    except Exception as e:
        logger.error(f"Failed to list LLMs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list LLMs: {str(e)}")

@app.post("/llms", response_model=MessageResponse)
def add_llm(request: AddLLMRequest):
    try:
        llm_manager.add_config(request.name, request.config.dict())
        return {"message": f"LLM {request.name} added successfully"}
    except Exception as e:
        logger.error(f"Failed to add LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add LLM: {str(e)}")

@app.delete("/llms/{name}", response_model=MessageResponse)
def remove_llm(name: str):
    try:
        llm_manager.remove_config(name)
        return {"message": f"LLM {name} removed successfully"}
    except Exception as e:
        logger.error(f"Failed to remove LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove LLM: {str(e)}")

@app.post("/llms/test", response_model=TestResponse)
def test_llm(request: TestModelRequest):
    try:
        config = llm_manager.get_config(request.name)
        if not config:
            raise HTTPException(status_code=404, detail=f"LLM {request.name} not found")
        ok, msg = llm_manager.test_model(config, "test")
        return {"success": ok, "message": msg}
    except Exception as e:
        logger.error(f"Failed to test LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test LLM: {str(e)}")

# Embedding Management endpoints
@app.get("/embeddings", response_model=EmbedListResponse)
def list_embeddings():
    try:
        return {"embeddings": embed_manager.get_names(), "selected": embed_manager.get_selected()}
    except Exception as e:
        logger.error(f"Failed to list embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list embeddings: {str(e)}")

@app.post("/embeddings", response_model=MessageResponse)
def add_embedding(request: AddEmbedRequest):
    try:
        embed_manager.add_config(request.name, request.config.dict())
        return {"message": f"Embedding {request.name} added successfully"}
    except Exception as e:
        logger.error(f"Failed to add embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add embedding: {str(e)}")

@app.delete("/embeddings/{name}", response_model=MessageResponse)
def remove_embedding(name: str):
    try:
        embed_manager.remove_config(name)
        return {"message": f"Embedding {name} removed successfully"}
    except Exception as e:
        logger.error(f"Failed to remove embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove embedding: {str(e)}")

@app.post("/embeddings/test", response_model=TestResponse)
def test_embedding(request: TestModelRequest):
    try:
        config = embed_manager.get_config(request.name)
        if not config:
            raise HTTPException(status_code=404, detail=f"Embedding {request.name} not found")
        ok, msg = embed_manager.test_model(request.name, "test", params=config)
        return {"success": ok, "message": msg}
    except Exception as e:
        logger.error(f"Failed to test embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test embedding: {str(e)}")

# Agent endpoints
@app.get("/agents", response_model=AgentListResponse)
def list_agents():
    try:
        return {"agents": agent_manager.get_available_agents()}
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

# File management endpoints
@app.get("/files", response_model=List[FileInfo])
def list_files():
    try:
        files = vector_store.get_all_files()
        return [
            FileInfo(
                filename=f['filename'],
                size=f.get('file_size'),
                type=f.get('file_type'),
                word_count=f.get('total_word_count'),
                upload_timestamp=f.get('upload_timestamp')
            ) for f in files
        ]
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.delete("/files/{filename}", response_model=MessageResponse)
def delete_file(filename: str):
    try:
        result = delete_selected_file(filename)
        # result is a tuple: (message, files_list, status)
        message = result[0] if isinstance(result, tuple) else result
        if "Successfully deleted" in message:
            return {"message": f"File {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=message)
    except Exception as e:
        logger.error(f"Failed to delete file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")