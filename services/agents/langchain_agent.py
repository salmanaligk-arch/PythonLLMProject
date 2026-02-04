"""
LangChain RAG Agent Implementation
"""
from langchain_classic.agents import initialize_agent, Tool, AgentType
from langchain_classic.memory import ConversationBufferMemory
from services.agent_tools import get_researcher_tool, get_writer_tool
from services.llm_manager import llm_manager

from typing import Dict, Any

class MultiProviderChatWrapper:
    """Factory wrapper to select any provider-supported chat LLM."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        provider = llm_config["provider"].lower()
        model = llm_config["model"]
        common_kwargs = {
            "model": model,
            "temperature": llm_config.get("temperature", 0),
            "timeout": llm_config.get("timeout"),
        }

        if provider in {"openai", "ollama", "lmstudio", "vllm"}:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                **common_kwargs,
                base_url=llm_config.get("base_url"),
                api_key=llm_config.get("api_key", "EMPTY"),
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(
                **common_kwargs,
                api_key=llm_config["api_key"],
            )

        elif provider in {"google", "gemini"}:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                **common_kwargs,
                google_api_key=llm_config["api_key"],
            )

        elif provider == "deepseek":
            # Example DeepSeek integration
            # Replace this with your actual DeepSeek SDK usage
            from langchain_deepseek import ChatDeepSeek

            self.llm = ChatDeepSeek(
                model=model,
                api_key=llm_config.get("api_key"),
                timeout=common_kwargs["timeout"]
            )
            
        elif provider in {"huggingface", "hf"}:
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

            hf_llm = HuggingFaceEndpoint(
                model=model,
                task="text-generation",
                huggingfacehub_api_token=llm_config.get("api_key"),
            )

            self.llm = ChatHuggingFace(llm=hf_llm)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

class LangChainRAGAgent:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Get LLM from chatbot (AI engine)
        #self.llm = LangchainLLM()
        llm_config = llm_manager.get_llm_config()
        wrapper = MultiProviderChatWrapper(llm_config)
        self.llm = wrapper.llm
        
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