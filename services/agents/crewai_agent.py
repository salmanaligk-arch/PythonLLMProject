"""
CrewAI RAG Agent Implementation with AI Engine Wrapper
"""

import os
from typing import Optional
from services.agent_tools import get_researcher_tool, get_writer_tool
from services.llm_manager import llm_manager
from services.logger import logger
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool


class CrewAIRAGAgent:
    def __init__(self, llm=None, verbose=True):
        self.verbose = verbose
        
        # Use a provided LLM or get config from AIEngine
        if llm:
            self.llm = llm
        else:
            llm_config = llm_manager.get_llm_config()
            self.llm = LLM(**llm_config)
        
        # Define tools
        @tool
        def research_tool(query: str) -> str:
            """Research information from the knowledge base."""
            return get_researcher_tool().research(query, None)
        
        @tool
        def writing_tool(content: str, format_prompt: str) -> str:
            """Format content according to requirements."""
            return get_writer_tool().write(content, format_prompt)
        
        # Create agents with our custom LLM
        self.researcher = Agent(
            role='Researcher',
            goal='Find relevant information',
            backstory='You are a skilled researcher.',
            tools=[research_tool],
            llm=self.llm,
            verbose=verbose,
            allow_delegation=False
        )
        
        self.writer = Agent(
            role='Writer',
            goal='Format content properly',
            backstory='You are a professional writer.',
            tools=[writing_tool],
            llm=self.llm,
            verbose=verbose,
            allow_delegation=False
        )

    def process(self, query: str, temp_document: str = None, format_prompt: str = None) -> str:
        if self.verbose:
            logger.info(f"🚢 CrewAI Agent: Processing query: {query}")
        
        # Enhance query if temp document provided
        enhanced_query = query
        if temp_document:
            enhanced_query = f"{query}\n\nContext: {temp_document}"
        
        # Create research task
        research_task = Task(
            description=f"Research: {enhanced_query}",
            agent=self.researcher,
            expected_output="Research findings"
        )
        
        tasks = [research_task]
        agents = [self.researcher]
        
        # Add writing task if format needed
        if format_prompt:
            writing_task = Task(
                description=f"Format the research results as: {format_prompt}",
                agent=self.writer,
                expected_output="Formatted content"
            )
            tasks.append(writing_task)
            agents.append(self.writer)
        
        # Create and run crew
        crew = Crew(agents=agents, tasks=tasks, verbose=self.verbose)
        result = crew.kickoff()
        
        if self.verbose:
            logger.info("🚢 CrewAI Agent: Completed")
        
        return str(result)