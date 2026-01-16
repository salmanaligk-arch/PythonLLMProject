import os
import warnings
from dotenv import load_dotenv
from api.routes import app
from frontend.ui import gui
import os
import threading
import uvicorn
import gradio as gr

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
load_dotenv("config.env")

if __name__=="__main__":
    def run_api():
        uvicorn.run(
            app,
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", 8000)),
        )

    def run_ui():
        # Always use soft theme
        selected_theme = gr.themes.Soft()

        gui.launch(
            server_name=os.getenv("UI_HOST", "0.0.0.0"),
            server_port=int(os.getenv("UI_PORT", 7860)),
            theme=selected_theme,
            share=os.getenv("SHARE_GRADIO", "false").lower() == "true",
        )

    threading.Thread(target=run_api).start()
    run_ui()