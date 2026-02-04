"""
LangChain RAG Agent Implementation with Native FAISS Integration
"""
import os
import pickle
from langchain_classic.agents import initialize_agent, Tool, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from services.llm_manager import llm_manager
from services.embed_manager import embed_manager
from services.logger import logger
from typing import Dict, Any, List, Optional

class LangChainEmbeddings:
    """LangChain compatible embeddings wrapper for our embed_manager."""
    
    def __init__(self):
        self.embed_manager = embed_manager
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        selected_model = self.embed_manager.get_selected()
        for text in texts:
            emb = self.embed_manager.get_embedding(text, selected_model)
            if emb is not None:
                embeddings.append(emb.tolist() if hasattr(emb, 'tolist') else list(emb))
            else:
                # Fallback to zero embedding if embedding fails
                dimension = int(os.getenv("VECTOR_DIMENSION", 768))
                embeddings.append([0.0] * dimension)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        selected_model = self.embed_manager.get_selected()
        emb = self.embed_manager.get_embedding(text, selected_model)
        if emb is not None:
            return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
        else:
            # Fallback to zero embedding if embedding fails
            dimension = int(os.getenv("VECTOR_DIMENSION", 768))
            return [0.0] * dimension

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
        llm_config = llm_manager.get_llm_config()
        wrapper = MultiProviderChatWrapper(llm_config)
        self.llm = wrapper.llm
        
        # Initialize LangChain compatible embeddings
        self.embeddings = LangChainEmbeddings()
        
        # Load existing FAISS index into LangChain FAISS vector store
        self.vector_store = None
        self.retriever = None
        self._load_faiss_vectorstore()
        
        # Create retrieval QA chain if vector store is available
        self.qa_chain = None
        if self.vector_store:
            self._setup_retrieval_chain()
        
        # Create enhanced tools with native retrieval
        tools = [
            Tool(
                name="faiss_research",
                description="Research information using native LangChain FAISS retrieval. Use this to find relevant information from the knowledge base.",
                func=self._native_faiss_research
            ),
            Tool(
                name="document_similarity_search",
                description="Perform similarity search on documents. Input should be a query string.",
                func=self._similarity_search
            ),
            Tool(
                name="retrieval_qa",
                description="Answer questions using retrieval-augmented generation. Input should be a question.",
                func=self._retrieval_qa
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
    
    def _load_faiss_vectorstore(self):
        """Load existing FAISS index into LangChain FAISS vector store."""
        try:
            index_path = os.getenv("VECTOR_INDEX_PATH", "faiss_indexes")
            faiss_file = os.path.join(index_path, "vector_index.faiss")
            pkl_file = os.path.join(index_path, "vector_index.pkl")
            
            if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                # Load documents and metadata
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    documents = data['documents']
                    metadata = data['metadata']
                
                # Convert to LangChain Document format
                langchain_docs = []
                for i, (doc_content, doc_meta) in enumerate(zip(documents, metadata)):
                    doc = Document(
                        page_content=doc_content,
                        metadata=doc_meta or {}
                    )
                    langchain_docs.append(doc)
                
                if langchain_docs:
                    # Load FAISS vector store from existing index
                    self.vector_store = FAISS.load_local(
                        folder_path=index_path,
                        embeddings=self.embeddings,
                        index_name="vector_index",
                        allow_dangerous_deserialization=True
                    )
                    
                    # Create retriever
                    self.retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": int(os.getenv("FAISS_SEARCH_K", 5))}
                    )
                    
                    if self.verbose:
                        logger.info(f"✅ Loaded LangChain FAISS vector store with {len(langchain_docs)} documents")
                else:
                    if self.verbose:
                        logger.warning("📝 No documents found in existing FAISS index")
            else:
                if self.verbose:
                    logger.warning("📁 No existing FAISS index found, will create new one when documents are added")
                    
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Failed to load FAISS vector store: {e}")
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval QA chain."""
        if not self.vector_store:
            return
            
        try:
            # Create custom prompt template
            prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=self.verbose
            )
            
            if self.verbose:
                logger.info("✅ Setup retrieval QA chain with custom prompt")
                
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Failed to setup retrieval chain: {e}")
    
    def _native_faiss_research(self, query: str) -> str:
        """Research using native LangChain FAISS integration."""
        if not self.vector_store:
            return "No vector store available. Please add documents first."
            
        try:
            # Use similarity search with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=int(os.getenv("FAISS_SEARCH_K", 5))
            )
            
            if not docs_with_scores:
                return "No relevant documents found for the query."
            
            # Format results with scores
            result = f"Found {len(docs_with_scores)} relevant document(s):\n\n"
            
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                filename = doc.metadata.get('filename', 'Unknown')
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                
                result += f"Document {i} (Score: {score:.3f}):\n"
                result += f"Source: {filename}\n"
                result += f"Content: {content_preview}\n\n"
            
            return result
            
        except Exception as e:
            return f"Research error: {str(e)}"
    
    def _similarity_search(self, query: str) -> str:
        """Perform similarity search and return formatted results."""
        if not self.vector_store:
            return "No vector store available for similarity search."
            
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            
            if not docs:
                return "No similar documents found."
            
            result = f"Similar documents for '{query}':\n\n"
            for i, doc in enumerate(docs, 1):
                filename = doc.metadata.get('filename', 'Unknown')
                result += f"{i}. {filename}: {doc.page_content[:150]}...\n\n"
            
            return result
            
        except Exception as e:
            return f"Similarity search error: {str(e)}"
    
    def _retrieval_qa(self, question: str) -> str:
        """Answer question using retrieval-augmented generation."""
        if not self.qa_chain:
            return "Retrieval QA chain not available. Please ensure documents are loaded."
            
        try:
            result = self.qa_chain({"query": question})
            answer = result.get('result', 'No answer generated')
            
            # Add source information if available
            source_docs = result.get('source_documents', [])
            if source_docs and self.verbose:
                sources = set(doc.metadata.get('filename', 'Unknown') for doc in source_docs)
                answer += f"\n\nSources: {', '.join(sources)}"
            
            return answer
            
        except Exception as e:
            return f"Retrieval QA error: {str(e)}"
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> bool:
        """Add new documents to the LangChain FAISS vector store."""
        try:
            if metadatas is None:
                metadatas = [{}] * len(texts)
            
            # Convert to Document objects
            documents = [Document(page_content=text, metadata=meta) 
                        for text, meta in zip(texts, metadatas)]
            
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)
            
            # Update retriever and QA chain
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": int(os.getenv("FAISS_SEARCH_K", 5))}
            )
            
            if not self.qa_chain:
                self._setup_retrieval_chain()
            
            # Save the updated index
            self._save_vectorstore()
            
            if self.verbose:
                logger.info(f"✅ Added {len(texts)} documents to LangChain FAISS vector store")
            
            return True
            
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    def _save_vectorstore(self):
        """Save the LangChain FAISS vector store."""
        if self.vector_store:
            try:
                index_path = os.getenv("VECTOR_INDEX_PATH", "faiss_indexes")
                os.makedirs(index_path, exist_ok=True)
                
                self.vector_store.save_local(
                    folder_path=index_path,
                    index_name="vector_index"
                )
                
                if self.verbose:
                    logger.info("💾 Saved LangChain FAISS vector store")
                    
            except Exception as e:
                if self.verbose:
                    logger.error(f"❌ Failed to save vector store: {e}")
    
    
    def process(self, query: str, temp_document: str = None, format_prompt: str = None) -> str:
        if self.verbose:
            print(f"🔗 LangChain Agent: Starting native FAISS retrieval for: {query}")
        
        try:
            # Handle temporary document by adding it to vector store
            if temp_document:
                temp_success = self.add_documents([temp_document], [{'filename': 'temp_session_doc', 'temporary': True}])
                if self.verbose and temp_success:
                    print("🔗 LangChain Agent: Added temporary document to FAISS vector store")
            
            # Use native retrieval QA for direct answers
            if self.qa_chain:
                if format_prompt:
                    formatted_query = f"{query}\n\nPlease format your response as: {format_prompt}"
                else:
                    formatted_query = query
                
                result = self._retrieval_qa(formatted_query)
                
                if self.verbose:
                    print("🔗 LangChain Agent: Native FAISS retrieval completed")
                
                return result
            else:
                # Fallback to agent-based approach if no QA chain available
                enhanced_query = query
                if temp_document:
                    enhanced_query = f"Query: {query}\n\nAdditional Context Document:\n{temp_document}"
                
                if format_prompt:
                    full_query = f"Please research: {enhanced_query}\n\nThen format the results as: {format_prompt}"
                else:
                    full_query = f"Please research: {enhanced_query}"
                
                result = self.agent.run(full_query)
                
                if self.verbose:
                    print("🔗 LangChain Agent: Agent execution completed (fallback mode)")
                
                return result
            
        except Exception as e:
            if self.verbose:
                print(f"🔗 LangChain Agent: Error - {str(e)}")
            
            # Final fallback
            return f"Error in native FAISS retrieval: {str(e)}"
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the loaded vector store."""
        if not self.vector_store:
            return {"status": "No vector store loaded", "documents": 0}
        
        try:
            # Get document count from FAISS index
            doc_count = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            
            return {
                "status": "Vector store loaded",
                "documents": doc_count,
                "embedding_dimension": int(os.getenv("VECTOR_DIMENSION", 768)),
                "search_k": int(os.getenv("FAISS_SEARCH_K", 5))
            }
        except Exception as e:
            return {"status": f"Error getting info: {e}", "documents": 0}