"""
LangChain RAG Agent Implementation
"""
from typing import Optional, List, Any, Mapping
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_classic.chat_models import init_chat_model
from services.agent_tools import get_researcher_tool, get_writer_tool
from services.chatbot import AIEngine
from services.llm_manager import llm_manager
# LangChain LLM Wrapper
from langchain.llms.base import LLM

class LangchainLLM(LLM):
    """LangChain-compatible LLM wrapper using our AI engine"""
    
    ai_engine: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ai_engine = AIEngine()
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        return self.ai_engine.call_ai(prompt)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"engine": "ai_engine_llm"}


class LangChainRAGAgent:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Get LLM from chatbot (AI engine)
        #self.llm = LangchainLLM()
        llm_config = llm_manager.get_llm_config()
        self.llm = init_chat_model(
            model_provider=llm_config["provider"],
            model=llm_config["model"],
            temperature=llm_config.get("temperature", 0),
            timeout=llm_config.get("timeout"),
            model_kwargs={
                "base_url": llm_config["base_url"].replace("/v1", "")
            }
        )
        
        # Create tools for the agent
        tools = [
            Tool(
                name="research",
                description="Research information from documents. Use this to find relevant information from the knowledge base.",
                func=lambda query: get_researcher_tool().research(query, None)
            ),
            Tool(
                name="write",
                description="Format and write information in a specific format. Input should be 'content|||format'",
                func=lambda input_str: get_writer_tool().write(*input_str.split('|||', 1)) if '|||' in input_str else "Invalid format. Use: content|||format"
            )
        ]
        
        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize agent using traditional API
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            handle_parsing_errors=True,
            verbose=verbose
        )
    
    def process(self, query: str, temp_document: str = None, format_prompt: str = None) -> str:
        if self.verbose:
            print(f"🔗 LangChain Agent: Starting agent execution for: {query}")
        
        try:
            # Handle temporary document by including it in the query
            enhanced_query = query
            if temp_document:
                enhanced_query = f"Query: {query}\n\nAdditional Context Document:\n{temp_document}"
                if self.verbose:
                    print("🔗 LangChain Agent: Enhanced query with temporary document")
            
            # Execute agent
            if format_prompt:
                full_query = f"Please research: {enhanced_query}\n\nThen format the results as: {format_prompt}"
            else:
                full_query = f"Please research: {enhanced_query}"
            
            result = self.agent.run(full_query)
            
            if self.verbose:
                print("🔗 LangChain Agent: Agent execution completed")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"🔗 LangChain Agent: Error - {str(e)}")
            # Fallback to direct tool usage
            research_result = get_researcher_tool().research(enhanced_query, temp_document)
            if format_prompt:
                return get_writer_tool().write(research_result, format_prompt)
            return research_result