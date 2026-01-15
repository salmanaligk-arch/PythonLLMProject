import os
import warnings
from dotenv import load_dotenv
from api.routes import app
from frontend.ui import gui
from services.settings_manager import settings_manager
import threading
import uvicorn
import gradio as gr

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

if __name__=="__main__":
    def run_api():
        uvicorn.run(
            app, 
            host=settings_manager.ui_config.api_host, 
            port=settings_manager.ui_config.api_port
        )

    def run_ui():
        # Always use soft theme
        selected_theme = gr.themes.Soft()
        
        gui.launch(
            server_name=settings_manager.ui_config.ui_host, 
            server_port=settings_manager.ui_config.ui_port, 
            theme=selected_theme,
            share=settings_manager.ui_config.share_gradio
        )

    threading.Thread(target=run_api).start()
    run_ui()