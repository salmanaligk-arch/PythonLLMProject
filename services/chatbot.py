"""
Central AI Engine for handling all AI interactions
This module provides the core AI functionality that agents use
"""

import os
from typing import Optional, List, Any, Mapping
from openai import OpenAI
from services.logger import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import settings manager
from services.settings_manager import settings_manager

class AIEngine:
    """Central AI engine for all AI interactions with online/offline fallback"""
    
    def __init__(self):
        # Get configuration from settings manager
        self.hf_token = settings_manager.llm_config.hf_token
        self.ollama_base_url = settings_manager.llm_config.ollama_base_url
        
        # Model configurations
        self.online_model = settings_manager.llm_config.hf_model
        self.offline_model = settings_manager.llm_config.ollama_model
        
        # Initialize LLM clients
        self._init_clients()
        
        # Test online availability
        self.online_available = self._test_online_connection()
        
        logger.info(f"AIEngine initialized - Online: {'✅' if self.online_available else '❌'}")
    
    def reload_config(self):
        """Reload configuration from settings manager and reinitialize"""
        logger.info("🔄 Reloading AI Engine configuration...")
        self.__init__()
        logger.info("✅ AI Engine configuration reloaded")
    
    def _init_clients(self):
        """Initialize online and offline LLM clients"""
        # Online LLM (HuggingFace)
        if self.hf_token and self.hf_token != "your_huggingface_token_here":
            self.online_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.hf_token,
                timeout=30,
                max_retries=2  # Reduce retries for faster fallback
            )
        else:
            self.online_client = None
            if not self.hf_token or self.hf_token == "your_huggingface_token_here":
                logger.warning("No valid HF_TOKEN found - online LLM disabled")
            else:
                logger.warning("HF_TOKEN is default placeholder - online LLM disabled")
        
        # Offline LLM (Ollama)
        self.offline_client = OpenAI(
            base_url=f"{self.ollama_base_url}/v1",
            api_key="ollama",  # Ollama doesn't need real API key
            timeout=1800,
            max_retries=1  # Reduce retries for local client
        )
    
    def _test_online_connection(self) -> bool:
        """Test if online LLM is available"""
        if not self.online_client or not self.hf_token:
            return False
            
        try:
            response = self.online_client.chat.completions.create(
                model=self.online_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10
            )
            return True
        except Exception as e:
            logger.warning(f"Online LLM test failed: {str(e)[:100]}...")
            return False
    
    def call_ai(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.1) -> str:
        """
        Main AI calling function with automatic online/offline fallback
        
        Args:
            prompt: The text prompt to send to the AI
            max_tokens: Maximum tokens in response (optional)
            temperature: Response randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            AI-generated response text
        """
        # Try online LLM first if available
        if self.online_available and self.online_client:
            try:
                logger.debug("Attempting to use online LLM.")
                return self._call_llm(self.online_client, self.online_model, prompt, max_tokens, temperature)
            except Exception as e:
                logger.warning(f"Online LLM failed: {str(e)[:100]}... - Falling back to offline")
        
        # Fallback to offline LLM
        try:
            logger.debug("Using offline LLM.")
            return self._call_llm(self.offline_client, self.offline_model, prompt, max_tokens, temperature)
        except Exception as e:
            error_msg = f"Both online and offline LLMs failed: {str(e)}"
            logger.error(error_msg)
            return f"AI Error: {error_msg}"
    
    def _call_llm(self, client: OpenAI, model: str, prompt: str, max_tokens: Optional[int], temperature: float) -> str:
        """Generic method to call an LLM."""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    
    def get_status(self) -> dict:
        """Get current status of AI engine"""
        return {
            "online_available": self.online_available,
            "online_model": self.online_model,
            "offline_model": self.offline_model,
            "hf_token_configured": bool(self.hf_token),
            "ollama_url": self.ollama_base_url
        }
    
    def refresh_online_status(self) -> bool:
        """Refresh online LLM availability status"""
        self.online_available = self._test_online_connection()
        logger.info(f"Online LLM status refreshed: {'✅' if self.online_available else '❌'}")
        return self.online_available

    def set_online_mode(self, enabled: bool) -> str:
        """Manually enable or disable online mode"""
        if enabled:
            if not self.hf_token:
                logger.warning("Cannot enable online mode: No HF_TOKEN configured")
                return "❌ Cannot enable online mode: No HF_TOKEN configured"
            # Test connection before enabling
            if self._test_online_connection():
                self.online_available = True
                logger.info("✅ Online mode enabled")
                return f"✅ Online mode enabled (using {self.online_model})"
            else:
                logger.warning("Cannot enable online mode: Connection test failed")
                return "❌ Cannot enable online mode: Connection test failed"
        else:
            self.online_available = False
            logger.info("⚠️ Online mode disabled - using Ollama")
            return f"⚠️ Online mode disabled (using Ollama: {self.offline_model})"

    def get_mode_status(self) -> str:
        """Get current mode status as a string"""
        if self.online_available:
            return f"🌐 Online Mode: {self.online_model}"
        else:
            return f"💻 Offline Mode: {self.offline_model} (Ollama)"

    def get_llm_config(self) -> dict:
        """
        Provides LLM configuration for CrewAI based on availability.
        Note: LiteLLM is used internally by CrewAI and is working correctly.
        """
        if self.online_available and self.online_client:
            logger.info("✅ Providing online LLM config for CrewAI (via LiteLLM)")
            # Use openai/ prefix with custom base_url for HuggingFace
            return {
                "model": f"openai/{self.online_model}",
                "base_url": "https://router.huggingface.co/v1",
                "api_key": self.hf_token,
                "temperature": settings_manager.llm_config.llm_temperature,
            }
        else:
            logger.info("⚠️ Providing offline LLM config for CrewAI (via LiteLLM)")
            # Use model name directly - Ollama's OpenAI-compatible endpoint
            # Increased timeout for larger models
            return {
                "model": self.offline_model,
                "base_url": f"{self.ollama_base_url}/v1",
                "api_key": "sk-no-key-required",
                "temperature": settings_manager.llm_config.llm_temperature,
                "timeout": settings_manager.llm_config.llm_timeout,
            }


# Global AI engine instance
ai_engine = AIEngine()

# Convenience functions for backward compatibility
def call_ai(prompt: str, **kwargs) -> str:
    """Direct AI call function"""
    return ai_engine.call_ai(prompt, **kwargs)

def get_ai_status() -> dict:
    """Get AI engine status"""
    return ai_engine.get_status()

def set_online_mode(enabled: bool) -> str:
    """Set online mode on/off"""
    return ai_engine.set_online_mode(enabled)

def get_mode_status() -> str:
    """Get current mode status"""
    return ai_engine.get_mode_status()


