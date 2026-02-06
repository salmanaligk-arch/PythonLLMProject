# Smart Assistant: A Modular RAG Application

This project is a sophisticated, modular Retrieval-Augmented Generation (RAG) application designed for interacting with your documents. It features a user-friendly web interface, a powerful backend, and a flexible architecture that supports multiple AI agents, language models, and embedding providers.

## ✨ Features

-   **Interactive Chat Interface**: Ask questions and get answers from your documents in a conversational format.
-   **Document Management**: Upload, browse, and delete various document types, including PDF, TXT, DOCX, and XLSX.
-   **Multi-Agent Support**: Choose from different AI agents (`simple`, `langchain`, `crewai`) to process your queries, each with unique strengths.
-   **Configurable AI Models**: Easily add, edit, and switch between multiple Large Language Models (LLMs) and embedding models through the UI.
-   **Dynamic Configuration**: Manage application settings, including chunk size, overlap, and default models, without restarting the server.
-   **Vector Storage**: Utilizes FAISS for efficient local vector storage and similarity search.
-   **Dual-Stack Application**: Combines a Gradio frontend for user interaction and a FastAPI backend for a robust API.
-   **Temporary Document Analysis**: Upload a document for a single session without permanently indexing it.

## 🏗️ Architecture

The application is built with a clear separation of concerns:

-   **Frontend ()**: A comprehensive user interface built with **Gradio**. It provides tabs for chatting, uploading files, browsing documents, and configuring settings.
-   **Backend ()**: A **FastAPI** application that exposes RESTful endpoints for all core functionalities like chat, file uploads, and RAG searches.
-   **Core Logic ()**: This is the heart of the application.
    -   ** & **: Manage the lifecycle and configuration of AI and embedding models, persisting settings to JSON files in the  directory.
    -   ****: Implements the FAISS-based vector store, handling document chunking, embedding, indexing, and retrieval.
    -   ****: An agent manager that dynamically loads and deploys different RAG agents (e.g., , ).
    -   ****: Contains the core  responsible for making calls to the selected LLM.
-   **Entry Point ()**: A simple script that launches both the FastAPI backend and the Gradio frontend in concurrent threads.

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

-   Python 3.8+
-   `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd PythonLLMProject
    ```

2.  **Create and activate a virtual environment:**
    -   **Windows:**
        ```bash
        python -m venv env
        .\env\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        python3 -m venv env
        source env/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The application uses a  file for base configuration. You can create this file in the root of the project to customize default settings.

```dotenv
# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
UI_HOST=0.0.0.0
UI_PORT=7860

# Vector Store Configuration
VECTOR_DIMENSION=768
VECTOR_INDEX_PATH=faiss_indexes

# File Processing
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# Agent Settings
DEFAULT_AGENT=simple
LLM_TIMEOUT=30
```

**Note:** For adding and managing AI models (like OpenAI, Ollama, or HuggingFace), use the **Settings** tab in the web interface. This is the recommended way to manage credentials and model endpoints.

### Running the Application

Execute the main script to launch both the UI and the API:

```bash
python Main.py
```

-   The Gradio Web UI will be available at `http://127.0.0.1:7860`.
-   The FastAPI backend API will be available at `http://127.0.0.1:8000`.

## 📖 Usage

1.  **Add Models**: Navigate to the **Settings** tab.
    -   Go to the **LLM** or **Embedding** sub-tabs.
    -   Fill in the form to add a new model (e.g., from Ollama or OpenAI). You will need to provide a name, provider, model name, and base URL/API key.
    -   Use the "Test" button to ensure the connection is working before saving.
2.  **Upload Documents**: Go to the **Upload Documents** tab.
    -   Select one or more files (`.pdf`, `.txt`, `.docx`, `.xlsx`).
    -   Choose the embedding model to use for indexing.
    -   Click **Upload & Index**.
3.  **Chat with Your Documents**: Go to the **Chat Search** tab.
    -   Select your desired AI Agent, LLM, and Embedding Model from the dropdowns.
    -   Type your question in the message box and press Enter or click **Send**.
    -   The agent will retrieve relevant information from your documents and generate a response.

## 📁 Project Structure

```
.
├── Main.py               # Application entry point
├── requirements.txt      # Project dependencies
├── config.env            # Environment variable configuration
├── api/
│   └── routes.py         # FastAPI backend routes
├── data/                 # Stores JSON configs for models and settings
├── db/
│   └── database.py       # SQLite database setup (for chat history)
├── frontend/
│   └── ui.py             # Gradio frontend interface
├── models/
│   └── schema.py         # Pydantic data models for API requests
├── services/
│   ├── agents/           # Implementations for different RAG agents
│   ├── file_handlers/    # Logic for parsing different document types
│   ├── agent_tools.py    # Tools used by agents
│   ├── chatbot.py        # Core AI engine for LLM interaction
│   ├── embed_manager.py  # Manages embedding models
│   ├── llm_manager.py    # Manages large language models
│   └── vector_store.py   # FAISS vector store implementation
└── faiss_indexes/        # Default location for stored FAISS indexes
```
