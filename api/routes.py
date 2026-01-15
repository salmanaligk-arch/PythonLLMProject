from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from models.schema import ChatRequest, RAGRequest
from services.chatbot import call_ai
from services.rag_agents import agent_manager
from services.vector_store import vector_store
from services.logger import logger
import os

app = FastAPI()

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return {"reply": call_ai(request.user_input)}

@app.post("/rag/search")
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

@app.post("/upload")
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
        
        return JSONResponse(
            content={
                "message": results,
                "successful_uploads": successful_uploads,
                "total_files": len(files)
            },
            status_code=200 if successful_uploads > 0 else 422
        )
    
    except Exception as e:
        logger.error(f"Upload batch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Multi-Agent RAG System API"}