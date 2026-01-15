# Multi-Agent RAG System

A comprehensive Retrieval Augmented Generation (RAG) system with multiple AI agent frameworks for document search and analysis.

## ✨ Features

- **Multiple Agent Frameworks**: Choose from Simple, LangChain, and CrewAI agents
- **Online/Offline Mode Toggle**: Switch between HuggingFace (online) and Ollama (offline)
- **FAISS Vector Storage**: Fast similarity search with local embedding storage (768-dim)
- **Ollama Integration**: Local embeddings using nomic-embed-text:v1.5
- **Document Upload**: Support for PDF and TXT files with automatic chunking
- **Document Deletion**: Remove files from FAISS index with automatic rebuild
- **Temporary Documents**: Session-only document analysis with enhanced retrieval
- **Custom Output Formatting**: Specify output format for structured responses
- **Web Interface**: User-friendly Gradio interface with 4 tabs including Settings
- **Settings Management**: Comprehensive configuration system with validation, export/import
- **REST API**: FastAPI backend for programmatic access

## 🤖 Agent Frameworks

### 1. Simple Agent
- Custom lightweight implementation
- Direct tool calls with AI engine
- Best for quick responses

### 2. LangChain Agent
- ReAct framework with conversation memory
- Extensive tool ecosystem
- Great for complex workflows

### 3. CrewAI Agent
- Multi-agent collaboration (Researcher + Writer)
- Role-based specialized agents
- Best for comprehensive research tasks
- Uses LiteLLM for model abstraction

## 🌐 Online/Offline Modes

### Online Mode (HuggingFace)
- Uses HuggingFace Inference API
- Model: Qwen/Qwen2.5-72B-Instruct (configurable)
- Requires HF_TOKEN in environment

### Offline Mode (Ollama)
- Uses local Ollama server
- Model: gpt-oss:20b (configurable via OLLAMA_MODEL)
- No internet required

## 🛠️ Tools Available to Agents

### Researcher Tool
- Searches existing vector database
- Enhanced search using temp document content
- Analyzes temporary documents (session-only)
- Combines information from multiple sources

### Writer Tool
- Formats research data according to user specifications
- Supports various output formats (bullet points, summary, table, etc.)
- Maintains consistency in formatting

## 📋 Prerequisites

- Python 3.10+
- Ollama installed locally (https://ollama.ai)
- HuggingFace API token

## 🚀 Installation & Setup

### 1. Create Virtual Environment

```bash
python -m venv env
# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file in project root:

```
HF_TOKEN=your_huggingface_token_here
HF_MODEL=Qwen/Qwen2.5-72B-Instruct
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
EMBEDDING_MODEL=nomic-embed-text:v1.5
```

### 4. Start Ollama

```bash
# Start Ollama service (keep running in background)
ollama serve

# In another terminal, pull required models
ollama pull nomic-embed-text:v1.5
ollama pull mistral:7b  # Optional fallback
```

## ▶️ Running the Application

### Start Web UI (Gradio)

```bash
python Main.py
```

The UI will be available at: `http://localhost:7860`

### Start REST API (FastAPI)

```bash
uvicorn api.routes:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

## 📖 Usage Guide

### Web UI (3 Tabs)

#### Tab 1: Search Documents (Main)
- **Agent Dropdown**: Select which agent to use
  - LangChain: Standard RAG with tools
  - SmolaGents: Lightweight agent framework
  - LlamaIndex: Specialized data indexing
  - CrewAI: Multi-agent collaboration
- **Query**: Ask questions about your documents
- **Response**: AI response based on relevant documents

#### Tab 2: Upload Documents
- **Upload Document**: Select .txt, .pdf, or .md files
- **Document Name**: Optional naming for reference
- **Upload & Index**: Process document and add to vector store
- **Database Management**: View count and clear database

#### Tab 3: File Browser  
- **Document Library**: View all uploaded files with metadata
- **File Content Viewer**: Preview document contents
- **Delete Management**: Remove individual files from the database

#### Tab 4: Settings (New!)
- **LLM Configuration**: Configure HuggingFace and Ollama models, API tokens, and LLM parameters
- **File Processing**: Set upload limits, chunk sizes, embedding models, and vector store settings
- **Agent Settings**: Configure default agent, search parameters, and verbosity
- **UI Settings**: Customize interface theme, server ports, and display options
- **Settings Management**: Save, reset, export, and import configuration
                        interactive=False,
                        wrap=True,
                        column_widths=["30%", "10%", "15%", "15%", "30%"]
                    )
                    
                    delete_status = gr.Textbox(
                        label="Delete Status",
                        interactive=False,
                        visible=True,
                        placeholder="Select a file and click Delete to remove it..."
                    )

### REST API Endpoints

#### Chat with Agent
```bash
POST /chat?agent=LangChain
Content-Type: application/json

{
  "user_input": "What is machine learning?"
}
```

#### Upload Document
```bash
POST /documents/upload?name=MyDocument
Content-Type: multipart/form-data

file: <your_file.txt>
```

#### Search Documents
```bash
GET /documents/search?query=machine%20learning&k=5
```

#### Find Related Documents
```bash
POST /documents/search-related
Content-Type: multipart/form-data

file: <your_file.txt>
```

#### Get Available Agents
```bash
GET /agents
```

#### Database Status
```bash
GET /documents/count
GET /health
```

#### Clear Database
```bash
DELETE /documents/clear
```

## 🏗️ Project Structure

```
PythonLLMProject/
├── Main.py              # Entry point
├── requirements.txt     # Dependencies
├── .env                 # Configuration (HF_TOKEN, OLLAMA_MODEL, etc.)
├── api/
│   └── routes.py        # FastAPI endpoints
├── db/
│   ├── database.py      # SQLite chat history
│   └── __init__.py
├── frontend/
│   └── ui.py            # Gradio interface (3 tabs)
├── models/
│   └── schema.py        # Pydantic models
├── services/
│   ├── agents/          # Agent implementations
│   │   ├── simple_agent.py     # Custom lightweight agent
│   │   ├── langchain_agent.py  # LangChain ReAct agent
│   │   ├── crewai_agent.py     # CrewAI multi-agent
│   │   └── __init__.py
│   ├── agent_tools.py   # Researcher & Writer tools
│   ├── chatbot.py       # AIEngine with online/offline modes
│   ├── vector_store.py  # FAISS vector store with delete support
│   ├── logger.py        # Logging
│   └── memory.py        # Chat history
├── faiss_indexes/       # FAISS index storage
└── data/                # Additional data storage
```

## 🔧 Configuration

### Models Used

- **Embedding Model**: `nomic-embed-text:v1.5` (768-dim, Ollama)
- **Online LLM**: `Qwen/Qwen2.5-72B-Instruct` (HuggingFace)
- **Offline LLM**: `gpt-oss:20b` (Ollama local, configurable)

### Key Parameters

**Vector Search**
- Default search results: 5 documents
- Similarity threshold: None (ranked by distance)
- Index type: FAISS L2 distance

**Agent Behavior**
- Temperature: 0.7
- Max tokens: 1024
- Context window: 4096 tokens

## 🔄 Workflow Examples

### Example 1: Upload and Query

1. Go to "Upload Documents" tab
2. Upload `report.txt` with company data
3. Go to "Search Documents" tab
4. Query: "What was the revenue this quarter?"
5. Agent searches documents and provides answer

### Example 2: Find Related Content

1. Go to "Find Related Documents" tab
2. Upload a new document
3. System finds similar documents in database
4. Use results to understand document relationships

## 📊 Agent Framework Comparison

| Feature | LangChain | SmolaGents | LlamaIndex | CrewAI |
|---------|-----------|-----------|-----------|---------|
| Complexity | Medium | Low | High | Medium |
| Performance | Good | Fast | Excellent | Good |
| Customization | High | Medium | High | Medium |
| Multi-agent | Limited | Limited | Limited | Yes |

## 🐛 Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check base URL in `.env`

### Embedding Model Not Found
```bash
ollama pull nomic-embed-text:v1.5
```

### Memory/Performance Issues
- Reduce search results (k parameter)
- Disable non-essential agents
- Use CPU-only FAISS version

### HuggingFace API Errors
- Verify HF_TOKEN in `.env`
- Check token permissions (needs API access)
- Ensure model access is granted

## 📝 Log Files

Application logs saved to: `genai.log`
Chat history saved to: `chat_history.db` (SQLite)

## 🚀 Performance Tips

1. **Batch uploads**: Add multiple documents at once
2. **Optimize queries**: Use specific keywords
3. **Agent selection**: LangChain is fastest, CrewAI is most thorough
4. **Vector store**: Periodically clear old documents

## 🔐 Security Notes

- **API Key**: Store HF_TOKEN in `.env` only, never commit
- **File uploads**: Validates file types (.txt, .pdf, .md)
- **Database**: SQLite `chat_history.db` stored locally
- **Vector store**: FAISS index stored locally

## 📚 Additional Resources

- [FAISS Documentation](https://faiss.ai/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [LangChain Docs](https://python.langchain.com/)
- [Gradio Guide](https://gradio.app/)

## ⚠️ Limitations

- Single-session vector store (data lost on restart unless saved)
- Embedding generation requires Ollama running
- Large documents may slow down search
- API rate limits from HuggingFace

## 🤝 Contributing

Feel free to extend with:
- More agent frameworks
- Alternative embedding models
- PostgreSQL/Redis backend
- Docker containerization
- Advanced search filters

## 📄 License

This project is provided as-is for educational and organizational use.

---

**Happy RAG searching! 🔍📚**
