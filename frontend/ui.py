import gradio as gr
import os
from services.chatbot import set_active_llm
from services.llm_manager import llm_manager
from services.file_handlers.temp_doc_handler import process_temp_document
from services.embed_manager import embed_manager
from services.rag_agents import agent_manager
# Import operations from separate modules
from frontend.file_operations import upload_documents, get_files_list, refresh_files, select_file, delete_selected_file
from frontend.settings_helpers import (
    save_general_settings,
    load_defaults,
    load_settings,
    get_llms_table,
    add_llm,
    get_llm_for_form,
    remove_llm,
    test_llm_with_values,
    get_embeds_table,
    add_embedding,
    run_test_embed,
    get_embed_for_form,
    remove_embedding,
)

# Load current settings (prefer `settings.json`, fallback to `default_settings.json`)
# Returns: llm, embed, agent, chunk_size, chunk_overlap, status_msg
llm_default, embed_default, agent_default, chunk_size_default_val, chunk_overlap_default_val, settings_status_msg = load_settings()

def update_fields_on_load():
    llm, embed, agent, chunk_size, chunk_overlap, status = load_settings()
    
    return (
        gr.update(value=llm or llm_manager.get_selected()),  # llm_dropdown
        gr.update(value=embed or embed_manager.get_selected()),  # chat_embed_dropdown
        gr.update(value=agent or os.getenv("DEFAULT_AGENT", "simple")),  # agent_dropdown
        gr.update(value=int(chunk_size or int(os.getenv("CHUNK_SIZE", 1000)))),  # chunk_size_slider
        gr.update(value=int(chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200)))),  # chunk_overlap_slider
        gr.update(value=embed or embed_manager.get_selected()),  # embed_dropdown
        gr.update(value=llm),  # general_llm_dropdown
        gr.update(value=embed),  # general_embed_dropdown
        gr.update(value=agent),  # general_agent_dropdown
        gr.update(value=int(chunk_size or int(os.getenv("CHUNK_SIZE", 1000)))),  # chunk_size_default
        gr.update(value=int(chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200)))),  # chunk_overlap_default
        status or ""  # settings_status
    )

def search_documents_chat(message, history, agent_type, embed_model, temp_document_file, format_prompt):
    # Handle temporary document file
    try:
        temp_document_content = process_temp_document(temp_document_file)
    except Exception as e:
        history.append({"role": "user", "content": "ERROR"})
        history.append({"role": "assistant", "content": str(e)})
        return history
    
    try:
        # If an embedding model was selected in the chat UI, set it globally
        if embed_model:
            try:
                embed_manager.set_selected(embed_model)
            except Exception:
                pass
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
                        value=lambda: load_settings()[2] or os.getenv("DEFAULT_AGENT", "simple"),
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
                        value=lambda: load_settings()[0] or llm_manager.get_selected(),
                        label="🔧 Select LLM",
                        elem_classes=["mode-toggle"]
                    )

                    # Embedding model selection for chat searches
                    chat_embed_dropdown = gr.Dropdown(
                        choices=embed_manager.get_names(),
                        value=lambda: load_settings()[1] or embed_manager.get_selected(),
                        label="🧩 Select Embedding Model"
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
            def respond_and_clear(message, history, agent_type, embed_model, temp_doc, format_prompt):
                new_history = search_documents_chat(message, history, agent_type, embed_model, temp_doc, format_prompt)
                return new_history, ""
            
            submit_btn.click(
                fn=respond_and_clear,
                inputs=[msg_input, chatbot, agent_dropdown, chat_embed_dropdown, temp_doc_input, format_prompt_input],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                fn=respond_and_clear,
                inputs=[msg_input, chatbot, agent_dropdown, chat_embed_dropdown, temp_doc_input, format_prompt_input],
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
                        value=embed_default or embed_manager.get_selected(),
                        label="🧩 Select Embedding Model"
                    )
                    chunk_size_slider = gr.Slider(minimum=200, maximum=5000, step=50, value=lambda: int(load_settings()[3] or int(os.getenv("CHUNK_SIZE", 1000))), label="🔪 Chunk Size")
                    chunk_overlap_slider = gr.Slider(minimum=0, maximum=2000, step=10, value=lambda: int(load_settings()[4] or int(os.getenv("CHUNK_OVERLAP", 200))), label="🔁 Chunk Overlap")
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
        
        # Settings Tab
        with gr.TabItem("⚙️ Settings", elem_id="settings_tab"):
            with gr.Tabs():
                # General
                with gr.TabItem("General"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### ⚙️ General Defaults")
                            general_llm_dropdown = gr.Dropdown(choices=llm_manager.get_names(), value=llm_default, label="Default LLM Model")
                            general_embed_dropdown = gr.Dropdown(choices=embed_manager.get_names(), value=embed_default, label="Default Embedding Model")
                            general_agent_dropdown = gr.Dropdown(choices=agent_manager.get_available_agents(), value=agent_default, label="Default Agent")
                            chunk_size_default = gr.Slider(minimum=200, maximum=5000, step=50, value=int(chunk_size_default_val or int(os.getenv("CHUNK_SIZE", 1000))), label="Default Chunk Size")
                            chunk_overlap_default = gr.Slider(minimum=0, maximum=2000, step=10, value=int(chunk_overlap_default_val or int(os.getenv("CHUNK_OVERLAP", 200))), label="Default Chunk Overlap")
                            save_defaults_btn = gr.Button("💾 Save Defaults", variant="primary")
                            load_defaults_btn = gr.Button("🔁 Load Defaults")
                        with gr.Column(scale=3):
                            gr.Markdown("### Status")
                            settings_status = gr.Textbox(lines=6, interactive=False, label="Save Status", value=settings_status_msg or "")

                            save_defaults_btn.click(fn=save_general_settings, inputs=[general_llm_dropdown, general_embed_dropdown, general_agent_dropdown, chunk_size_default, chunk_overlap_default], outputs=[settings_status, llm_dropdown, embed_dropdown, agent_dropdown])
                            load_defaults_btn.click(fn=load_defaults, inputs=[], outputs=[general_llm_dropdown, general_embed_dropdown, general_agent_dropdown, chunk_size_default, chunk_overlap_default, settings_status])

                # LLM
                with gr.TabItem("LLM"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 🔧 Registered LLMs")
                            llms_table = gr.Dataframe(headers=["Name", "Provider", "Model"], value=get_llms_table(), interactive=False)
                            llm_select_dropdown = gr.Dropdown(choices=llm_manager.get_names(), value=llm_manager.get_selected(), label="Select LLM")
                            llm_action_status = gr.Textbox(lines=3, interactive=False, label="Status")
                        with gr.Column(scale=3):
                            gr.Markdown("### ➕ Add / Edit LLM")
                            new_llm_name = gr.Textbox(label="Name")
                            new_llm_provider = gr.Textbox(label="Provider", placeholder="e.g. openai, ollama")
                            new_llm_model = gr.Textbox(label="Model")
                            new_llm_base = gr.Textbox(label="Base URL")
                            new_llm_api_key = gr.Textbox(label="API Key", type="password")
                            new_llm_timeout = gr.Slider(minimum=1, maximum=2000, step=1, value=30, label="Timeout (s)")
                            new_llm_retries = gr.Slider(minimum=0, maximum=10, step=1, value=2, label="Max Retries")
                            new_llm_temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1, label="Temperature")
                            with gr.Row():
                                add_llm_btn = gr.Button("➕ Add / Update LLM", variant="primary")
                                test_current_llm_btn = gr.Button("▶️ Test Current")
                                delete_llm_btn = gr.Button("🗑️ Delete Selected", variant="stop")

                    def select_llm(row_or_name):
                        if not row_or_name:
                            return "", "openai", "", "", "", 30, 2, 0.1, ""
                        if isinstance(row_or_name, list):
                            name = row_or_name[0]
                        else:
                            name = row_or_name
                        form = get_llm_for_form(name)
                        return (*form, name)

                    llm_select_dropdown.change(fn=select_llm, inputs=llm_select_dropdown, outputs=[new_llm_name, new_llm_provider, new_llm_model, new_llm_base, new_llm_api_key, new_llm_timeout, new_llm_retries, new_llm_temperature])

                    add_llm_btn.click(fn=add_llm, inputs=[new_llm_name, new_llm_provider, new_llm_model, new_llm_base, new_llm_api_key, new_llm_timeout, new_llm_retries, new_llm_temperature], outputs=[llms_table, llm_dropdown, llm_action_status])
                    test_current_llm_btn.click(fn=test_llm_with_values, inputs=[new_llm_name, new_llm_provider, new_llm_model, new_llm_base, new_llm_api_key, new_llm_timeout, new_llm_retries, new_llm_temperature], outputs=[llm_action_status])
                    delete_llm_btn.click(fn=remove_llm, inputs=[llm_select_dropdown], outputs=[llms_table, llm_action_status, llm_select_dropdown, llm_dropdown])

                # Embedding
                with gr.TabItem("Embedding"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 🧩 Registered Embedding Models")
                            embeds_table = gr.Dataframe(headers=["Name", "Provider", "Model"], value=get_embeds_table(), interactive=False)
                            embed_select_dropdown = gr.Dropdown(choices=embed_manager.get_names(), value=embed_manager.get_selected(), label="Select Embedding")
                            embed_action_status = gr.Textbox(lines=3, interactive=False, label="Status")
                        with gr.Column(scale=3):
                            gr.Markdown("### ➕ Add / Edit Embedding Model")
                            new_embed_name = gr.Textbox(label="Name")
                            new_embed_provider = gr.Textbox(label="Provider", placeholder="e.g. ollama, huggingface")
                            new_embed_model = gr.Textbox(label="Model")
                            new_embed_url = gr.Textbox(label="URL")
                            new_embed_api_key = gr.Textbox(label="API Key", type="password")
                            with gr.Row():
                                add_embed_btn = gr.Button("➕ Add / Update Embedding", variant="primary")
                                test_current_embed_btn = gr.Button("▶️ Test Current")
                                delete_embed_btn = gr.Button("🗑️ Delete Selected", variant="stop")

                    def select_embed(row_or_name):
                        if not row_or_name:
                            return "", "ollama", "", "", ""
                        if isinstance(row_or_name, list):
                            name = row_or_name[0]
                        else:
                            name = row_or_name
                        name, provider, model, api_key, url = get_embed_for_form(name)
                        return name, provider, model, api_key, url, name

                    embed_select_dropdown.change(fn=select_embed, inputs=embed_select_dropdown, outputs=[new_embed_name, new_embed_provider, new_embed_model, new_embed_api_key, new_embed_url])

                    add_embed_btn.click(fn=add_embedding, inputs=[new_embed_name, new_embed_provider, new_embed_model, new_embed_api_key, new_embed_url], outputs=[embeds_table, embed_dropdown, embed_action_status])
                    test_current_embed_btn.click(fn=run_test_embed, inputs=[new_embed_name, gr.Textbox(value="test", visible=False), new_embed_provider, new_embed_model, new_embed_api_key, new_embed_url, gr.Textbox(value="", visible=False)], outputs=[embed_action_status])
                    delete_embed_btn.click(fn=remove_embedding, inputs=[embed_select_dropdown], outputs=[embeds_table, embed_action_status, embed_select_dropdown, embed_dropdown])
    
    # Instructions
    with gr.Accordion("📚 How to Use", open=False):
           gr.Markdown("""
           ### Quick Start

           1. **Upload & Index**: Open "Upload Documents", select files, choose an embedding model and chunk settings, then click **Upload & Index**.
           2. **Chat with Documents**: Go to "Chat Search", pick an agent and LLM, optionally attach a temporary document, then ask a question.
           3. **Browse Files**: Use "File Browser" to view uploaded documents and inspect their content.

           ### Configuration
           - **Defaults**: Set app-wide defaults under Settings → General.
           - **Add Models**: Add or update LLMs and Embedding models under Settings → LLM / Embedding.
           - **Test Models**: Use the ▶️ Test buttons to validate LLM and embedding endpoints before saving.

           ### Examples
           - Ask: "Summarize the uploaded PDF in 5 bullet points."
           - Ask with format: "List the key findings as a table" (use the Output Format field).

           ### Troubleshooting
           - If embedding tests fail, verify the embedding **URL**, **model**, and **API key** in Settings → Embedding.
           - If LLM requests time out, increase the timeout in Settings → LLM or check the base URL and API key.
           - Check the Upload Status box for indexing errors.

           If you'd like this section shortened or expanded, tell me what to include.
           """)
    gui.load(
        fn=update_fields_on_load,
        inputs=[],
        outputs=[
            llm_dropdown,
            chat_embed_dropdown,
            agent_dropdown,
            chunk_size_slider,
            chunk_overlap_slider,
            embed_dropdown,
            general_llm_dropdown,
            general_embed_dropdown,
            general_agent_dropdown,
            chunk_size_default,
            chunk_overlap_default,
            settings_status
        ]
    )
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