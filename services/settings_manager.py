"""
Settings Manager for Multi-Agent RAG System
Handles configuration loading, saving, and validation
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv, set_key, dotenv_values
from services.logger import logger

@dataclass
class LLMConfig:
    """LLM Configuration settings"""
    hf_token: str = "your_huggingface_token_here"  # Default, can be overridden via .env or UI
    hf_model: str = "Qwen/Qwen2.5-72B-Instruct"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:8b"
    crewai_tracing_enabled: bool = False
    llm_timeout: int = 30
    llm_max_retries: int = 2
    llm_temperature: float = 0.1

@dataclass
class FileProcessingConfig:
    """File Processing Configuration"""
    max_file_size: str = "50MB"
    allowed_file_types: str = ".pdf,.txt,.docx,.xlsx"
    max_chunk_size: int = 1000
    chunk_overlap: int = 100
    vector_dimension: int = 768
    vector_index_path: str = "faiss_indexes"
    embedding_model: str = "nomic-embed-text:v1.5"

@dataclass
class AgentConfig:
    """Agent Configuration"""
    default_agent: str = "simple"
    agent_verbose: bool = True
    faiss_search_k: int = 5
    max_search_results: int = 5

@dataclass
class UIConfig:
    """UI Configuration"""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860
    ui_theme: str = "soft"  # Fixed to soft theme
    chat_height: int = 400
    max_chat_history: int = 50
    share_gradio: bool = False

@dataclass
class LoggingConfig:
    """Logging Configuration"""
    log_level: str = "INFO"
    log_file: str = "genai.log"
    debug_mode: bool = False

class SettingsManager:
    """Central settings manager for the application"""
    
    def __init__(self, config_file: str = "config.env"):
        self.config_file = config_file
        self.settings_file = "settings.json"
        
        # Default configurations
        self.llm_config = LLMConfig()
        self.file_config = FileProcessingConfig()
        self.agent_config = AgentConfig()
        self.ui_config = UIConfig()
        self.logging_config = LoggingConfig()
        
        # Load settings
        self.load_settings()
        logger.info("Settings Manager initialized")

    def load_settings(self):
        """Load settings from config file and custom settings"""
        try:
            # Load from .env file first (for sensitive data like tokens)
            load_dotenv(".env")
            # Then load from config template file
            load_dotenv(self.config_file)
            self._load_from_env()
            
            # Load custom settings if they exist (highest priority)
            if os.path.exists(self.settings_file):
                self._load_from_json()
                
            logger.info("✅ Settings loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading settings: {e}")

    def _load_from_env(self):
        """Load settings from environment variables"""
        # LLM Settings - Priority: Environment > Default
        self.llm_config.hf_token = os.getenv("HF_TOKEN", self.llm_config.hf_token)
        self.llm_config.hf_model = os.getenv("HF_MODEL", self.llm_config.hf_model)
        self.llm_config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.llm_config.ollama_base_url)
        self.llm_config.ollama_model = os.getenv("OLLAMA_MODEL", self.llm_config.ollama_model)
        self.llm_config.crewai_tracing_enabled = os.getenv("CREWAI_TRACING_ENABLED", "false").lower() == "true"
        self.llm_config.llm_timeout = int(os.getenv("LLM_TIMEOUT", self.llm_config.llm_timeout))
        self.llm_config.llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", self.llm_config.llm_max_retries))
        self.llm_config.llm_temperature = float(os.getenv("LLM_TEMPERATURE", self.llm_config.llm_temperature))

        # File Processing Settings
        self.file_config.max_file_size = os.getenv("MAX_FILE_SIZE", self.file_config.max_file_size)
        self.file_config.allowed_file_types = os.getenv("ALLOWED_FILE_TYPES", self.file_config.allowed_file_types)
        self.file_config.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", self.file_config.max_chunk_size))
        self.file_config.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.file_config.chunk_overlap))
        self.file_config.vector_dimension = int(os.getenv("VECTOR_DIMENSION", self.file_config.vector_dimension))
        self.file_config.vector_index_path = os.getenv("VECTOR_INDEX_PATH", self.file_config.vector_index_path)
        self.file_config.embedding_model = os.getenv("EMBEDDING_MODEL", self.file_config.embedding_model)

        # Agent Settings
        self.agent_config.default_agent = os.getenv("DEFAULT_AGENT", self.agent_config.default_agent)
        self.agent_config.agent_verbose = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
        self.agent_config.faiss_search_k = int(os.getenv("FAISS_SEARCH_K", self.agent_config.faiss_search_k))
        self.agent_config.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", self.agent_config.max_search_results))

        # UI Settings
        self.ui_config.api_host = os.getenv("API_HOST", self.ui_config.api_host)
        self.ui_config.api_port = int(os.getenv("API_PORT", self.ui_config.api_port))
        self.ui_config.ui_host = os.getenv("UI_HOST", self.ui_config.ui_host)
        self.ui_config.ui_port = int(os.getenv("UI_PORT", self.ui_config.ui_port))
        self.ui_config.ui_theme = os.getenv("UI_THEME", self.ui_config.ui_theme)
        self.ui_config.chat_height = int(os.getenv("CHAT_HEIGHT", self.ui_config.chat_height))
        self.ui_config.max_chat_history = int(os.getenv("MAX_CHAT_HISTORY", self.ui_config.max_chat_history))
        self.ui_config.share_gradio = os.getenv("SHARE_GRADIO", "false").lower() == "true"

        # Logging Settings
        self.logging_config.log_level = os.getenv("LOG_LEVEL", self.logging_config.log_level)
        self.logging_config.log_file = os.getenv("LOG_FILE", self.logging_config.log_file)
        self.logging_config.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

    def _load_from_json(self):
        """Load custom settings from JSON file"""
        try:
            with open(self.settings_file, 'r') as f:
                custom_settings = json.load(f)
                
            # Update configurations with custom settings
            if 'llm_config' in custom_settings:
                self._update_dataclass(self.llm_config, custom_settings['llm_config'])
            if 'file_config' in custom_settings:
                self._update_dataclass(self.file_config, custom_settings['file_config'])
            if 'agent_config' in custom_settings:
                self._update_dataclass(self.agent_config, custom_settings['agent_config'])
            if 'ui_config' in custom_settings:
                self._update_dataclass(self.ui_config, custom_settings['ui_config'])
            if 'logging_config' in custom_settings:
                self._update_dataclass(self.logging_config, custom_settings['logging_config'])
                
        except Exception as e:
            logger.error(f"Error loading custom settings: {e}")

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass object with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            settings_data = {
                'llm_config': asdict(self.llm_config),
                'file_config': asdict(self.file_config),
                'agent_config': asdict(self.agent_config),
                'ui_config': asdict(self.ui_config),
                'logging_config': asdict(self.logging_config)
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)
                
            logger.info("✅ Settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving settings: {e}")
            return False

    def update_env_file(self, key: str, value: str):
        """Update a specific key in the .env file"""
        try:
            set_key(self.config_file, key, str(value))
            return True
        except Exception as e:
            logger.error(f"Error updating {key} in .env file: {e}")
            return False

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings"""
        return {
            'llm_config': asdict(self.llm_config),
            'file_config': asdict(self.file_config),
            'agent_config': asdict(self.agent_config),
            'ui_config': asdict(self.ui_config),
            'logging_config': asdict(self.logging_config)
        }

    def update_llm_settings(self, settings: Dict[str, Any]) -> bool:
        """Update LLM settings"""
        try:
            # If HF_TOKEN is being updated, also update the .env file for persistence
            if 'hf_token' in settings and settings['hf_token'].strip():
                self.update_env_file('HF_TOKEN', settings['hf_token'])
            
            self._update_dataclass(self.llm_config, settings)
            return self.save_settings()
        except Exception as e:
            logger.error(f"Error updating LLM settings: {e}")
            return False

    def update_file_processing_settings(self, settings: Dict[str, Any]) -> bool:
        """Update file processing settings"""
        try:
            self._update_dataclass(self.file_config, settings)
            return self.save_settings()
        except Exception as e:
            logger.error(f"Error updating file processing settings: {e}")
            return False

    def update_agent_settings(self, settings: Dict[str, Any]) -> bool:
        """Update agent settings"""
        try:
            self._update_dataclass(self.agent_config, settings)
            return self.save_settings()
        except Exception as e:
            logger.error(f"Error updating agent settings: {e}")
            return False

    def update_ui_settings(self, settings: Dict[str, Any]) -> bool:
        """Update UI settings"""
        try:
            self._update_dataclass(self.ui_config, settings)
            return self.save_settings()
        except Exception as e:
            logger.error(f"Error updating UI settings: {e}")
            return False

    def reset_to_defaults(self):
        """Reset all settings to default values"""
        self.llm_config = LLMConfig()
        self.file_config = FileProcessingConfig()
        self.agent_config = AgentConfig()
        self.ui_config = UIConfig()
        self.logging_config = LoggingConfig()
        
        # Remove custom settings file
        if os.path.exists(self.settings_file):
            os.remove(self.settings_file)
            
        logger.info("Settings reset to defaults")
        return True

    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, str]:
        """Validate settings and return any errors"""
        errors = {}
        
        # Validate LLM settings
        if 'llm_config' in settings:
            llm = settings['llm_config']
            if 'llm_timeout' in llm and (llm['llm_timeout'] < 10 or llm['llm_timeout'] > 300):
                errors['llm_timeout'] = "Timeout must be between 10 and 300 seconds"
            if 'llm_temperature' in llm and (llm['llm_temperature'] < 0 or llm['llm_temperature'] > 2):
                errors['llm_temperature'] = "Temperature must be between 0.0 and 2.0"
                
        # Validate file settings  
        if 'file_config' in settings:
            file_cfg = settings['file_config']
            if 'max_chunk_size' in file_cfg and (file_cfg['max_chunk_size'] < 100 or file_cfg['max_chunk_size'] > 5000):
                errors['max_chunk_size'] = "Chunk size must be between 100 and 5000 characters"
            if 'vector_dimension' in file_cfg and (file_cfg['vector_dimension'] < 128 or file_cfg['vector_dimension'] > 2048):
                errors['vector_dimension'] = "Vector dimension must be between 128 and 2048"
                
        # Validate UI settings
        if 'ui_config' in settings:
            ui_cfg = settings['ui_config']
            if 'api_port' in ui_cfg and (ui_cfg['api_port'] < 1000 or ui_cfg['api_port'] > 65535):
                errors['api_port'] = "Port must be between 1000 and 65535"
            if 'ui_port' in ui_cfg and (ui_cfg['ui_port'] < 1000 or ui_cfg['ui_port'] > 65535):
                errors['ui_port'] = "Port must be between 1000 and 65535"
                
        return errors

    def export_settings(self, filepath: str) -> bool:
        """Export settings to a file"""
        try:
            settings = self.get_all_settings()
            with open(filepath, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            return False

    def import_settings(self, filepath: str) -> bool:
        """Import settings from a file"""
        try:
            with open(filepath, 'r') as f:
                settings = json.load(f)
            
            # Update all configurations
            if 'llm_config' in settings:
                self._update_dataclass(self.llm_config, settings['llm_config'])
            if 'file_config' in settings:
                self._update_dataclass(self.file_config, settings['file_config'])
            if 'agent_config' in settings:
                self._update_dataclass(self.agent_config, settings['agent_config'])
            if 'ui_config' in settings:
                self._update_dataclass(self.ui_config, settings['ui_config'])
            if 'logging_config' in settings:
                self._update_dataclass(self.logging_config, settings['logging_config'])
            
            return self.save_settings()
        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            return False

# Global settings manager instance
settings_manager = SettingsManager()