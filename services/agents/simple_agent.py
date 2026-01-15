"""
Simple RAG Agent Implementation
"""

import os
from services.agent_tools import get_researcher_tool, get_writer_tool
from services.chatbot import ai_engine

class SimpleRAGAgent:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def process(self, query: str, temp_document: str = None, format_prompt: str = None) -> str:
        if self.verbose:
            print(f"⚙️ Simple Agent: Processing query: {query}")
        
        research_result = get_researcher_tool().research(query, temp_document)
        
        if format_prompt:
            if self.verbose:
                print(f"⚙️ Simple Agent: Formatting with: {format_prompt}")
            final_result = get_writer_tool().write(research_result, format_prompt)
            return final_result
        else:
            if self.verbose:
                print("⚙️ Simple Agent: Returning direct results")
            return research_result