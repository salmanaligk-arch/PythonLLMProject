import gradio as gr
import os
from services.vector_store import vector_store
from services.chatbot import ai_engine, set_active_llm, get_mode_status
from services.llm_manager import llm_manager
from services.file_handlers.temp_doc_handler import process_temp_document
from services.embed_manager import embed_manager
from services.rag_agents import agent_manager
# Import operations from separate modules
from frontend.file_operations import upload_documents, get_files_list, refresh_files, select_file, delete_selected_file

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
                        value=os.getenv("DEFAULT_AGENT", "simple"),
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
                    
                    # LLM Selection Dropdown (replaces checkbox toggle)
                    llm_dropdown = gr.Dropdown(
                        choices=llm_manager.get_names(),
                        value=llm_manager.get_selected(),
                        label="🔧 Select LLM",
                        elem_classes=["mode-toggle"]
                    )

                    def change_llm(name: str):
                        # Set active LLM and return status message
                        return set_active_llm(name)

                    llm_dropdown.change(
                        fn=change_llm,
                        inputs=llm_dropdown,
                        outputs=[],
                    )
                
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="💬 Chat with Documents",
                        height=int(os.getenv("CHAT_HEIGHT", 400))
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
                    embed_dropdown = gr.Dropdown(
                        choices=embed_manager.get_names(),
                        value=embed_manager.get_selected(),
                        label="🧩 Select Embedding Model"
                    )
                    chunk_size_slider = gr.Slider(minimum=200, maximum=5000, step=50, value=int(os.getenv("CHUNK_SIZE", 1000)), label="🔪 Chunk Size")
                    chunk_overlap_slider = gr.Slider(minimum=0, maximum=2000, step=10, value=int(os.getenv("CHUNK_OVERLAP", 200)), label="🔁 Chunk Overlap")
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
                inputs=[file_upload, embed_dropdown, chunk_size_slider, chunk_overlap_slider],
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