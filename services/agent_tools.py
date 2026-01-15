import os
from typing import Dict, Any, List, Optional
from services.vector_store import vector_store
from services.logger import logger
from services.settings_manager import settings_manager

class ResearcherTool:
    def __init__(self):
        # Import here to avoid circular imports
        from services.chatbot import ai_engine
        self.ai_engine = ai_engine
    
    def research(self, query: str, temp_document: Optional[str] = None) -> str:
        logger.info(f"Research tool called with query: {query}")
        
        # Build enhanced search query - include temp document content for better retrieval
        search_query = query
        if temp_document:
            # Extract key terms from temp document to enhance search
            # Use first 500 chars to avoid making query too long
            temp_excerpt = temp_document[:500] if len(temp_document) > 500 else temp_document
            search_query = f"{query} {temp_excerpt}"
        
        # Search existing vector database with enhanced query
        max_results = settings_manager.agent_config.max_search_results
        search_results = vector_store.search(search_query, k=max_results)
        
        context = ""
        if search_results:
            context += "\n\nRelevant Documents from Knowledge Base:\n"
            for result in search_results:
                context += f"- {result['content']}...\n"
        
        # Add temporary document as additional context
        if temp_document:
            context += f"\n\nUploaded Document Content:\n{temp_document}\n"
        
        prompt = f"""
Based on the following context, answer the query: {query}

{context}

Provide a comprehensive answer based on the available information.
"""
        
        # Use AI engine for AI interaction
        response = self.ai_engine.call_ai(prompt)
        logger.info("Research completed using AI engine")
        return response

class WriterTool:
    def __init__(self):
        # Import here to avoid circular imports
        from services.chatbot import ai_engine
        self.ai_engine = ai_engine
    
    def write(self, research_data: str, format_prompt: str) -> str:
        logger.info(f"Writer tool called with format: {format_prompt}")
        
        prompt = f"""
Research Data:
{research_data}

Please write the above research data in the following format:
{format_prompt}

Ensure the output follows the specified format exactly.
"""
        
        # Use AI engine for AI interaction
        response = self.ai_engine.call_ai(prompt)
        logger.info("Writing completed using AI engine")
        return response

# Global tool instances - lazy initialization
_researcher_tool = None
_writer_tool = None

def get_researcher_tool():
    global _researcher_tool
    if _researcher_tool is None:
        _researcher_tool = ResearcherTool()
    return _researcher_tool

def get_writer_tool():
    global _writer_tool
    if _writer_tool is None:
        _writer_tool = WriterTool()
    return _writer_tool

# For backward compatibility
researcher_tool = get_researcher_tool
writer_tool = get_writer_tool