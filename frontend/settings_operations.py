"""
Settings Operations for the UI
Handles all settings-related functionality including save, load, export, import, and validation
"""

from services.settings_manager import settings_manager
from services.chatbot import ai_engine
from services.logger import logger

def validate_settings(llm_settings, file_settings, agent_settings, ui_settings):
    """Validate settings before saving"""
    errors = {}
    
    # Validate LLM settings
    if llm_settings.get('llm_timeout', 0) < 10:
        errors['llm_timeout'] = "Timeout must be at least 10 seconds"
    
    if llm_settings.get('llm_max_retries', 0) < 1:
        errors['llm_max_retries'] = "Max retries must be at least 1"
    
    if not (0.0 <= llm_settings.get('llm_temperature', 0.1) <= 2.0):
        errors['llm_temperature'] = "Temperature must be between 0.0 and 2.0"
    
    # Validate file processing settings
    if file_settings.get('max_chunk_size', 0) < 100:
        errors['max_chunk_size'] = "Chunk size must be at least 100 characters"
    
    if file_settings.get('chunk_overlap', 0) < 0:
        errors['chunk_overlap'] = "Chunk overlap cannot be negative"
    
    if file_settings.get('vector_dimension', 0) < 128:
        errors['vector_dimension'] = "Vector dimension must be at least 128"
    
    # Validate agent settings
    if agent_settings.get('faiss_search_k', 0) < 1:
        errors['faiss_search_k'] = "Search K must be at least 1"
    
    if agent_settings.get('max_search_results', 0) < 1:
        errors['max_search_results'] = "Max search results must be at least 1"
    
    # Validate UI settings
    if not (1000 <= ui_settings.get('api_port', 8000) <= 65535):
        errors['api_port'] = "API port must be between 1000 and 65535"
    
    if not (1000 <= ui_settings.get('ui_port', 7860) <= 65535):
        errors['ui_port'] = "UI port must be between 1000 and 65535"
    
    if ui_settings.get('chat_height', 0) < 200:
        errors['chat_height'] = "Chat height must be at least 200 pixels"
    
    return errors

def save_all_settings(*args):
    """Save all settings"""
    try:
        # Update all configurations
        llm_settings = {
            'hf_token': args[0],
            'hf_model': args[1], 
            'ollama_base_url': args[2],
            'ollama_model': args[3],
            'llm_timeout': int(args[4]),
            'llm_max_retries': int(args[5]),
            'llm_temperature': float(args[6]),
            'crewai_tracing_enabled': args[7]
        }
        
        file_settings = {
            'max_file_size': args[8],
            'allowed_file_types': args[9],
            'max_chunk_size': int(args[10]),
            'chunk_overlap': int(args[11]),
            'vector_dimension': int(args[12]),
            'embedding_model': args[13],
            'vector_index_path': args[14]
        }
        
        agent_settings = {
            'default_agent': args[15],
            'agent_verbose': args[16],
            'faiss_search_k': int(args[17]),
            'max_search_results': int(args[18])
        }
        
        ui_settings = {
            'api_host': args[19],
            'api_port': int(args[20]),
            'ui_host': args[21],
            'ui_port': int(args[22]),
            'ui_theme': 'soft',  # Fixed to soft theme
            'chat_height': int(args[23]),
            'max_chat_history': int(args[24]),
            'share_gradio': args[25]
        }
        
        # Validate settings
        validation_errors = validate_settings(llm_settings, file_settings, agent_settings, ui_settings)
        if validation_errors:
            error_messages = [f"{field}: {msg}" for field, msg in validation_errors.items()]
            return "❌ Validation errors:\n" + "\n".join(error_messages)
        
        # Save all settings
        settings_manager.update_llm_settings(llm_settings)
        settings_manager.update_file_processing_settings(file_settings)
        settings_manager.update_agent_settings(agent_settings)
        settings_manager.update_ui_settings(ui_settings)
        
        # Reload AI engine if LLM settings changed (especially HF token)
        ai_engine.reload_config()
        
        # Check if theme was changed
        if ui_settings.get('ui_theme') != settings_manager.ui_config.ui_theme:
            return "✅ Settings saved successfully! AI Engine reloaded. ⚠️ Please restart the application to apply theme changes.", settings_manager.agent_config.default_agent
        
        return "✅ Settings saved successfully! AI Engine reloaded with new configuration.", settings_manager.agent_config.default_agent
        
    except Exception as e:
        from services.settings_manager import AgentConfig
        return f"❌ Error saving settings: {str(e)}", AgentConfig().default_agent

def reset_all_settings():
    """Reset all settings to defaults"""
    try:
        settings_manager.reset_to_defaults()
        
        # Return default values for all inputs (excluding theme since it's fixed)
        from services.settings_manager import LLMConfig, FileProcessingConfig, AgentConfig, UIConfig
        
        defaults = [
            LLMConfig().hf_token, LLMConfig().hf_model, LLMConfig().ollama_base_url, LLMConfig().ollama_model,
            LLMConfig().llm_timeout, LLMConfig().llm_max_retries, LLMConfig().llm_temperature, LLMConfig().crewai_tracing_enabled,
            FileProcessingConfig().max_file_size, FileProcessingConfig().allowed_file_types, FileProcessingConfig().max_chunk_size,
            FileProcessingConfig().chunk_overlap, FileProcessingConfig().vector_dimension, FileProcessingConfig().embedding_model,
            FileProcessingConfig().vector_index_path,
            AgentConfig().default_agent, AgentConfig().agent_verbose, AgentConfig().faiss_search_k, AgentConfig().max_search_results,
            UIConfig().api_host, UIConfig().api_port, UIConfig().ui_host, UIConfig().ui_port,
            UIConfig().chat_height, UIConfig().max_chat_history, UIConfig().share_gradio,
            "🔄 Settings reset to defaults successfully!"
        ]
        
        return (*defaults, AgentConfig().default_agent)
        
    except Exception as e:
        from services.settings_manager import AgentConfig
        return (*([None] * 26 + [f"❌ Error resetting settings: {str(e)}"]), AgentConfig().default_agent)

def export_settings():
    """Export current settings"""
    try:
        export_path = "exported_settings.json"
        settings_manager.export_settings(export_path)
        return f"✅ Settings exported to {export_path}"
    except Exception as e:
        return f"❌ Error exporting settings: {str(e)}"

def import_settings(file):
    """Import settings from file"""
    try:
        if file is None:
            return "❌ No file selected for import"
        
        settings_manager.import_settings(file.name)
        # Reload AI engine with imported settings
        ai_engine.reload_config()
        return "✅ Settings imported successfully! AI Engine reloaded with new configuration."
    except Exception as e:
        return f"❌ Error importing settings: {str(e)}"

def reload_current_settings():
    """Reload current settings into the UI"""
    try:
        settings_manager.load_settings()
        
        # Return current values for all inputs (excluding theme since it's fixed)
        current_values = [
            settings_manager.llm_config.hf_token, settings_manager.llm_config.hf_model, 
            settings_manager.llm_config.ollama_base_url, settings_manager.llm_config.ollama_model,
            settings_manager.llm_config.llm_timeout, settings_manager.llm_config.llm_max_retries, 
            settings_manager.llm_config.llm_temperature, settings_manager.llm_config.crewai_tracing_enabled,
            settings_manager.file_config.max_file_size, settings_manager.file_config.allowed_file_types, 
            settings_manager.file_config.max_chunk_size, settings_manager.file_config.chunk_overlap, 
            settings_manager.file_config.vector_dimension, settings_manager.file_config.embedding_model,
            settings_manager.file_config.vector_index_path,
            settings_manager.agent_config.default_agent, settings_manager.agent_config.agent_verbose, 
            settings_manager.agent_config.faiss_search_k, settings_manager.agent_config.max_search_results,
            settings_manager.ui_config.api_host, settings_manager.ui_config.api_port, 
            settings_manager.ui_config.ui_host, settings_manager.ui_config.ui_port,
            settings_manager.ui_config.chat_height, settings_manager.ui_config.max_chat_history, 
            settings_manager.ui_config.share_gradio,
            "🔄 Settings reloaded successfully!"
        ]
        
        return (*current_values, settings_manager.agent_config.default_agent)
        
    except Exception as e:
        from services.settings_manager import AgentConfig
        return (*([None] * 26 + [f"❌ Error reloading settings: {str(e)}"]), AgentConfig().default_agent)