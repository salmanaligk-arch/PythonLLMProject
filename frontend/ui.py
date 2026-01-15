import gradio as gr
import os
from services.vector_store import vector_store
from services.chatbot import ai_engine, set_online_mode, get_mode_status
from services.file_handlers.temp_doc_handler import process_temp_document
from services.rag_agents import agent_manager
from services.settings_manager import settings_manager

# Import operations from separate modules
from frontend.file_operations import upload_documents, get_files_list, refresh_files, select_file, delete_selected_file
from frontend.settings_operations import (
    save_all_settings, reset_all_settings, export_settings, 
    import_settings, reload_current_settings
)

def search_documents_chat(message, history, agent_type, temp_document_file, format_prompt):
    # Handle temporary document file
    try:
        temp_document_content = process_temp_document(temp_document_file)
    except Exception as e:
        history.append({"role": "user", "content": "ERROR"})
        history.append({"role": "assistant", "content": str(e)})
        return history
    
    try:
        # Create agent using AgentManager
        agent = agent_manager.create_agent(agent_type)
        
        result = agent.process(message, temp_document_content, format_prompt)
        
        # Format the response
        response = result
        if format_prompt:
            response = f"**Formatted as {format_prompt}:**\\n\\n{result}"
            
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history
    except Exception as e:
        error_message = str(e)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {error_message}"})
        return history


# Create the interface with enhanced layout
with gr.Blocks(title="Smart Assistant") as gui:
    gr.Markdown("# 🤖 Smart Assistant")
    gr.Markdown("## Advanced document search with AI agents")
    
    with gr.Tabs() as tabs:
        # Chat Tab (Main Search)
        with gr.TabItem("💬 Chat Search", elem_id="chat_tab"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Agent selection
                    agent_dropdown = gr.Dropdown(
                        choices=agent_manager.get_available_agents(),
                        value=settings_manager.agent_config.default_agent,
                        label="🤖 Select AI Agent"
                    )
                    
                    # Temporary document upload
                    temp_doc_input = gr.File(
                        file_types=[".txt", ".pdf", ".docx", ".xlsx"],
                        file_count="single",
                        label="📎 Temporary Document (Optional)"
                    )
                    
                    # Format prompt (optional)
                    format_prompt_input = gr.Textbox(
                        lines=2,
                        placeholder="Optional: Specify output format (e.g., 'bullet points', 'summary', 'table')",
                        label="📝 Output Format (Optional)"
                    )
                    
                    # Online/Offline Mode Toggle
                    def get_toggle_label():
                        if ai_engine.online_available:
                            return "🌐 Online Mode (HuggingFace)"
                        else:
                            return "💻 Offline Mode (Ollama)"
                    
                    online_toggle = gr.Checkbox(
                        label=get_toggle_label(),
                        value=ai_engine.online_available,
                        elem_classes=["mode-toggle"]
                    )
                    
                    def toggle_online_mode(enabled):
                        set_online_mode(enabled)
                        if enabled:
                            return gr.update(label="🌐 Online Mode (HuggingFace)")
                        else:
                            return gr.update(label="💻 Offline Mode (Ollama)")
                    
                    online_toggle.change(
                        fn=toggle_online_mode,
                        inputs=online_toggle,
                        outputs=online_toggle
                    )
                
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="💬 Chat with Documents",
                        height=settings_manager.ui_config.chat_height
                    )
                    
                    # Message input
                    msg_input = gr.Textbox(
                        placeholder="Ask questions about your documents...",
                        label="💭 Your Message",
                        lines=2
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Send", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear Chat")
            
            # Chat functionality
            def respond_and_clear(message, history, agent_type, temp_doc, format_prompt):
                new_history = search_documents_chat(message, history, agent_type, temp_doc, format_prompt)
                return new_history, ""
            
            submit_btn.click(
                fn=respond_and_clear,
                inputs=[msg_input, chatbot, agent_dropdown, temp_doc_input, format_prompt_input],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                fn=respond_and_clear,
                inputs=[msg_input, chatbot, agent_dropdown, temp_doc_input, format_prompt_input],
                outputs=[chatbot, msg_input]
            )
            
            clear_btn.click(
                lambda: [],
                outputs=chatbot
            )
        
        # Upload Tab
        with gr.TabItem("📁 Upload Documents", elem_id="upload_tab"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📤 Upload Documents")
                    file_upload = gr.File(
                        file_types=[".txt", ".pdf", ".docx", ".xlsx"],
                        file_count="multiple",
                        label="📄 Select Documents (PDF, TXT, DOCX, XLSX)"
                    )
                    upload_btn = gr.Button("⬆️ Upload & Index", variant="primary", size="lg")
                
                with gr.Column():
                    gr.Markdown("### 📊 Upload Status")
                    upload_output = gr.Textbox(
                        lines=8,
                        label="Status",
                        interactive=False,
                        placeholder="Upload status will appear here..."
                    )
            
            upload_btn.click(
                fn=upload_documents,
                inputs=file_upload,
                outputs=upload_output
            )
        
        # File Browser Tab
        with gr.TabItem("🗂️ File Browser", elem_id="browser_tab"):
            selected_file = gr.State(value=None)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Document Library")
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                        delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                    
                    files_table = gr.Dataframe(
                        headers=["📄 File Name", "📝 Type", "📏 Size", "🔤 Words", "📅 Uploaded"],
                        datatype=["str", "str", "str", "str", "str"],
                        value=get_files_list(),
                        label="Click on a file to view its content",
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
                
                with gr.Column(scale=3):
                    gr.Markdown("### 👁️ File Content Viewer")
                    
                    # Show currently selected file
                    selected_file_display = gr.Textbox(
                        label="📁 Currently Selected File",
                        interactive=False,
                        placeholder="No file selected",
                        value=""
                    )
                    
                    file_content = gr.Textbox(
                        lines=20,
                        label="Document Content",
                        placeholder="Click on a file in the table to view its content...",
                        interactive=False,
                        max_lines=50
                    )
            
            # File browser functionality
            refresh_btn.click(
                fn=refresh_files,
                outputs=files_table
            )
            
            files_table.select(
                fn=select_file,
                outputs=[file_content, selected_file, selected_file_display]
            )
            
            delete_btn.click(
                fn=delete_selected_file,
                inputs=selected_file,
                outputs=[delete_status, files_table, file_content]
            )
        
        # Settings Tab
        with gr.TabItem("⚙️ Settings", elem_id="settings_tab"):
            with gr.Tabs() as settings_tabs:
                # LLM Configuration
                with gr.TabItem("🤖 LLM Configuration"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🌐 Online LLM (HuggingFace)")
                            hf_token_input = gr.Textbox(
                                value=settings_manager.llm_config.hf_token,
                                label="🔑 HuggingFace Token",
                                type="password"
                            )
                            hf_model_input = gr.Textbox(
                                value=settings_manager.llm_config.hf_model,
                                label="🤖 HuggingFace Model",
                                placeholder="e.g., Qwen/Qwen2.5-72B-Instruct"
                            )
                            
                        with gr.Column():
                            gr.Markdown("### 💻 Local LLM (Ollama)")
                            ollama_url_input = gr.Textbox(
                                value=settings_manager.llm_config.ollama_base_url,
                                label="🌐 Ollama Base URL",
                                placeholder="http://localhost:11434"
                            )
                            ollama_model_input = gr.Textbox(
                                value=settings_manager.llm_config.ollama_model,
                                label="🤖 Ollama Model",
                                placeholder="e.g., deepseek-r1:8b"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### ⚙️ LLM Parameters")
                            llm_timeout_input = gr.Number(
                                value=settings_manager.llm_config.llm_timeout,
                                label="⏱️ Timeout (seconds)",
                                minimum=10,
                                maximum=300
                            )
                            llm_retries_input = gr.Number(
                                value=settings_manager.llm_config.llm_max_retries,
                                label="🔄 Max Retries",
                                minimum=1,
                                maximum=10
                            )
                            
                        with gr.Column():
                            llm_temperature_input = gr.Slider(
                                value=settings_manager.llm_config.llm_temperature,
                                minimum=0.0,
                                maximum=2.0,
                                step=0.1,
                                label="🌡️ Temperature"
                            )
                            crewai_tracing_input = gr.Checkbox(
                                value=settings_manager.llm_config.crewai_tracing_enabled,
                                label="📊 CrewAI Tracing Enabled"
                            )
                
                # File Processing Configuration  
                with gr.TabItem("📁 File Processing"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📤 Upload Settings")
                            max_file_size_input = gr.Textbox(
                                value=settings_manager.file_config.max_file_size,
                                label="📏 Max File Size",
                                placeholder="50MB"
                            )
                            allowed_types_input = gr.Textbox(
                                value=settings_manager.file_config.allowed_file_types,
                                label="📄 Allowed File Types",
                                placeholder=".pdf,.txt,.docx,.xlsx"
                            )
                            
                        with gr.Column():
                            gr.Markdown("### 🔪 Text Chunking")
                            max_chunk_size_input = gr.Number(
                                value=settings_manager.file_config.max_chunk_size,
                                label="📝 Max Chunk Size (characters)",
                                minimum=100,
                                maximum=5000
                            )
                            chunk_overlap_input = gr.Number(
                                value=settings_manager.file_config.chunk_overlap,
                                label="🔄 Chunk Overlap (characters)",
                                minimum=0,
                                maximum=500
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🔍 Vector Store")
                            vector_dimension_input = gr.Number(
                                value=settings_manager.file_config.vector_dimension,
                                label="📐 Vector Dimension",
                                minimum=128,
                                maximum=2048
                            )
                            embedding_model_input = gr.Textbox(
                                value=settings_manager.file_config.embedding_model,
                                label="🧠 Embedding Model",
                                placeholder="nomic-embed-text:v1.5"
                            )
                            
                        with gr.Column():
                            vector_index_path_input = gr.Textbox(
                                value=settings_manager.file_config.vector_index_path,
                                label="📂 Vector Index Path",
                                placeholder="faiss_indexes"
                            )
                
                # Agent Configuration
                with gr.TabItem("🤖 Agent Settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🎭 Agent Behavior")
                            default_agent_input = gr.Dropdown(
                                choices=["simple", "langchain", "crewai"],
                                value=settings_manager.agent_config.default_agent,
                                label="🤖 Default Agent"
                            )
                            agent_verbose_input = gr.Checkbox(
                                value=settings_manager.agent_config.agent_verbose,
                                label="📢 Verbose Output"
                            )
                            
                        with gr.Column():
                            gr.Markdown("### 🔍 Search Settings")
                            faiss_search_k_input = gr.Number(
                                value=settings_manager.agent_config.faiss_search_k,
                                label="🔍 FAISS Search K",
                                minimum=1,
                                maximum=20
                            )
                            max_search_results_input = gr.Number(
                                value=settings_manager.agent_config.max_search_results,
                                label="📊 Max Search Results",
                                minimum=1,
                                maximum=50
                            )
                
                # UI Configuration
                with gr.TabItem("🎨 UI Settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🌐 Server Settings")
                            api_host_input = gr.Textbox(
                                value=settings_manager.ui_config.api_host,
                                label="🖥️ API Host",
                                placeholder="0.0.0.0"
                            )
                            api_port_input = gr.Number(
                                value=settings_manager.ui_config.api_port,
                                label="🚪 API Port",
                                minimum=1000,
                                maximum=65535
                            )
                            
                        with gr.Column():
                            ui_host_input = gr.Textbox(
                                value=settings_manager.ui_config.ui_host,
                                label="🖥️ UI Host",
                                placeholder="0.0.0.0"
                            )
                            ui_port_input = gr.Number(
                                value=settings_manager.ui_config.ui_port,
                                label="🚪 UI Port",
                                minimum=1000,
                                maximum=65535
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🎨 Interface Settings")
                            gr.Markdown("🎨 **Theme**: Soft Theme (Fixed)")
                            chat_height_input = gr.Number(
                                value=settings_manager.ui_config.chat_height,
                                label="💬 Chat Height (px)",
                                minimum=200,
                                maximum=800
                            )
                            
                        with gr.Column():
                            max_chat_history_input = gr.Number(
                                value=settings_manager.ui_config.max_chat_history,
                                label="📚 Max Chat History",
                                minimum=10,
                                maximum=500
                            )
                            share_gradio_input = gr.Checkbox(
                                value=settings_manager.ui_config.share_gradio,
                                label="🌍 Share Gradio Interface"
                            )
            
            # Settings Actions
            with gr.Row():
                save_settings_btn = gr.Button("💾 Save Settings", variant="primary", size="lg")
                reset_settings_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
                load_settings_btn = gr.Button("📥 Load Saved Settings", variant="secondary")
                export_settings_btn = gr.Button("📤 Export Settings", variant="secondary")
                import_settings_btn = gr.Button("📥 Import Settings", variant="secondary")
            
            # Settings status
            settings_status = gr.Textbox(
                label="Settings Status",
                interactive=False,
                placeholder="Settings actions will show status here..."
            )
            
            # Settings file upload for import
            settings_file_upload = gr.File(
                file_types=[".json"],
                label="📁 Settings File (for import)",
                visible=False
            )
            
            
            # Connect settings buttons
            all_inputs = [
                hf_token_input, hf_model_input, ollama_url_input, ollama_model_input,
                llm_timeout_input, llm_retries_input, llm_temperature_input, crewai_tracing_input,
                max_file_size_input, allowed_types_input, max_chunk_size_input, chunk_overlap_input,
                vector_dimension_input, embedding_model_input, vector_index_path_input,
                default_agent_input, agent_verbose_input, faiss_search_k_input, max_search_results_input,
                api_host_input, api_port_input, ui_host_input, ui_port_input,
                chat_height_input, max_chat_history_input, share_gradio_input
            ]
            
            # Settings action buttons
            with gr.Row():
                # Buttons already defined above
                pass
            
            save_settings_btn.click(
                fn=save_all_settings,
                inputs=all_inputs,
                outputs=[settings_status, agent_dropdown]
            )
            
            reset_settings_btn.click(
                fn=reset_all_settings,
                outputs=all_inputs + [settings_status, agent_dropdown]
            )
            
            load_settings_btn.click(
                fn=reload_current_settings,
                outputs=all_inputs + [settings_status, agent_dropdown]
            )
            
            export_settings_btn.click(
                fn=export_settings,
                outputs=settings_status
            )
            
            import_settings_btn.click(
                lambda: gr.update(visible=True),
                outputs=settings_file_upload
            )
            
            settings_file_upload.upload(
                fn=import_settings,
                inputs=settings_file_upload,
                outputs=settings_status
            )
            
            # Settings Help Section
            with gr.Accordion("❓ Settings Help", open=False):
                gr.Markdown("""
                ### 🤖 LLM Configuration
                - **HuggingFace Token**: Your API token for accessing HuggingFace models
                - **HuggingFace Model**: The online model to use (e.g., Qwen/Qwen2.5-72B-Instruct)
                - **Ollama Base URL**: Local Ollama server URL (usually http://localhost:11434)
                - **Ollama Model**: Local model name (e.g., deepseek-r1:8b)
                - **Temperature**: Controls response randomness (0.0 = deterministic, 2.0 = very creative)
                - **Timeout**: Maximum seconds to wait for LLM response
                - **Max Retries**: Number of retry attempts for failed requests
                
                ### 📁 File Processing
                - **Max File Size**: Maximum allowed file upload size (e.g., 50MB)
                - **Allowed File Types**: Comma-separated file extensions (.pdf,.txt,.docx,.xlsx)
                - **Max Chunk Size**: Maximum characters per text chunk for processing
                - **Chunk Overlap**: Characters overlap between consecutive chunks
                - **Vector Dimension**: Embedding vector size (must match embedding model)
                - **Embedding Model**: Ollama model for generating embeddings
                - **Vector Index Path**: Folder to store FAISS indexes
                
                ### 🤖 Agent Settings
                - **Default Agent**: Which agent to select by default (simple, langchain, crewai)
                - **Verbose Output**: Enable detailed logging from agents
                - **FAISS Search K**: Number of similar documents to retrieve
                - **Max Search Results**: Maximum results to show from search
                
                ### 🎨 UI Settings
                - **API/UI Host**: Server host address (0.0.0.0 for all interfaces)
                - **API/UI Port**: Server port numbers
                - **UI Theme**: Visual theme (default, soft, monochrome, glass)
                - **Chat Height**: Height of chat window in pixels
                - **Max Chat History**: Maximum messages to keep in chat
                - **Share Gradio**: Create public link to share interface
                
                ### 💡 Tips
                - Save settings after changes to persist them
                - Some changes (like ports/hosts) require restart
                - Export settings to backup your configuration
                - Reset to defaults if you encounter issues
                """)
    
    # Instructions
    with gr.Accordion("📚 How to Use", open=False):
        gr.Markdown("""
        ### 🚀 Getting Started:
        
        1. **📁 Upload Documents**: Go to "Upload Documents" tab and add your PDF/TXT files
        2. **💬 Chat Search**: Use natural language to search and ask questions about your documents
        3. **🗂️ File Browser**: View all uploaded files and their content
        
        ### 🤖 AI Agents:
        - **Simple**: Custom lightweight agent using direct tool calls
        - **LangChain**: Real LangChain agent with ReAct framework and conversation memory
        - **CrewAI**: Real CrewAI multi-agent crew with specialized roles
        
        ### 💡 Tips:
        - Use specific questions for better results
        - Upload related documents for comprehensive answers
        - Try different agents to compare responses
        - Use output format to structure responses (e.g., "bullet points", "table", "summary")
        """)

# Launch with custom CSS for better appearance
css = """
#chat_tab, #upload_tab, #browser_tab, #settings_tab {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gradio-container {
    max-width: 1400px !important;
}

/* Settings tab styling */
#settings_tab .gradio-row {
    margin: 10px 0;
}

#settings_tab .gradio-column {
    padding: 15px;
}

#settings_tab .gradio-markdown h3 {
    color: #2563eb;
    margin-bottom: 10px;
}

/* Settings buttons styling */
.settings-actions .gradio-button {
    margin: 5px;
    border-radius: 8px;
}

/* Settings status box */
.settings-status {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px;
}
"""

gui.css = css

def create_interface():
    """Function to create and return the Gradio interface"""
    return gui