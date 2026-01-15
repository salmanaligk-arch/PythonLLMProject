"""
Agent Manager - Centralized management of all RAG agents
This module handles agent loading, creation, and coordination
"""

from typing import Dict, Any
from services.logger import logger

class AgentManager:
    """Manager for all RAG agents"""
    
    def __init__(self):
        self.agents = {}
        self._load_agents()
    
    def _load_agents(self):
        """Dynamically load available agents"""
        try:
            from services.agents.simple_agent import SimpleRAGAgent
            self.agents['simple'] = SimpleRAGAgent
            logger.info("✓ Simple agent loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load Simple agent: {e}")
        
        try:
            from services.agents.langchain_agent import LangChainRAGAgent
            self.agents['langchain'] = LangChainRAGAgent
            logger.info("✓ LangChain agent loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load LangChain agent: {e}")
        
        try:
            from services.agents.crewai_agent import CrewAIRAGAgent
            self.agents['crewai'] = CrewAIRAGAgent
            logger.info("✓ CrewAI agent loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load CrewAI agent: {e}")
    
    def get_available_agents(self) -> list:
        """Get list of available agent types"""
        return list(self.agents.keys())
    
    def create_agent(self, agent_type: str, verbose: bool = True):
        """Create an agent instance of the specified type"""
        if agent_type not in self.agents:
            available = ", ".join(self.get_available_agents())
            raise ValueError(f"Agent type '{agent_type}' not available. Available: {available}")
        
        return self.agents[agent_type](verbose=verbose)
    
    def process_query(self, query: str, agent_type: str = 'simple', temp_document: str = None, 
                     format_prompt: str = None, verbose: bool = True) -> str:
        """
        Process a query using the specified agent
        
        Args:
            query: User's question or request
            agent_type: Type of agent to use (simple, langchain, llamaindex, crewai)
            temp_document: Optional temporary document content
            format_prompt: Optional formatting instructions
            verbose: Whether to show verbose output
            
        Returns:
            Agent's response as string
        """
        try:
            agent = self.create_agent(agent_type, verbose=verbose)
            return agent.process(query, temp_document, format_prompt)
        except Exception as e:
            error_msg = f"Error with {agent_type} agent: {str(e)}"
            logger.error(error_msg)
            
            # Fallback to simple agent if requested agent fails
            if agent_type != 'simple' and 'simple' in self.agents:
                logger.info("Falling back to simple agent...")
                try:
                    fallback_agent = self.create_agent('simple', verbose=verbose)
                    return fallback_agent.process(query, temp_document, format_prompt)
                except Exception as fallback_error:
                    return f"All agents failed. Last error: {str(fallback_error)}"
            
            return error_msg

# Global agent manager instance
agent_manager = AgentManager()

# Agent Factory (backward compatibility)
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str, verbose: bool = True):
        return agent_manager.create_agent(agent_type, verbose)
    
    @staticmethod
    def get_available_agents():
        return agent_manager.get_available_agents()

# Convenience functions
def get_available_agents():
    return agent_manager.get_available_agents()

def create_agent(agent_type: str, verbose: bool = True):
    return agent_manager.create_agent(agent_type, verbose)

def process_query(query: str, agent_type: str = 'simple', temp_document: str = None, 
                 format_prompt: str = None, verbose: bool = True) -> str:
    return agent_manager.process_query(query, agent_type, temp_document, format_prompt, verbose)