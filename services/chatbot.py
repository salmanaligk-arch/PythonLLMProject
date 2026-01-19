"""
Central AI Engine for handling all AI interactions
This module provides the core AI functionality that agents use
"""

import os
from typing import Optional, List, Any, Mapping
from openai import OpenAI
from services.logger import logger
from dotenv import load_dotenv
from services.llm_manager import llm_manager
import os

# Load environment variables from .env and config.env
load_dotenv()
load_dotenv("config.env")

class AIEngine:
    """Central AI engine for all AI interactions with online/offline fallback"""
    
    def __init__(self):
        # Use the external llm_manager to manage available LLMs
        self.llm_manager = llm_manager

        # LLM tuning from environment
        self.llm_timeout = int(os.getenv("LLM_TIMEOUT", 30))
        self.llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", 2))
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.1))

        # Initialize active client based on llm_manager selection
        self._init_active_client()

        logger.info(f"AIEngine initialized - Active LLM: {self.llm_manager.get_selected()}")
    
    def reload_config(self):
        """Reload configuration from settings manager and reinitialize"""
        logger.info("🔄 Reloading AI Engine configuration...")
        self.__init__()
        logger.info("✅ AI Engine configuration reloaded")
    
    def _init_active_client(self):
        """Create the OpenAI-compatible client for the currently selected LLM."""
        selected = self.llm_manager.get_selected()
        if not selected:
            logger.warning("No LLM selected in LLMManager")
            self.active_client = None
            self.active_name = None
            self.active_model = None
            return

        cfg = self.llm_manager.get_config(selected)
        self.active_name = selected
        self.active_model = cfg.get("model") if cfg else None
        self.active_client = self.llm_manager.create_client(selected)
    
    def _test_active_llm(self) -> bool:
        """Quick smoke test for the currently active LLM client."""
        if not self.active_client or not self.active_model:
            return False
        try:
            _ = self.active_client.chat.completions.create(
                model=self.active_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10,
            )
            return True
        except Exception as e:
            logger.warning(f"Active LLM test failed: {str(e)[:100]}...")
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
        # Use the active LLM only. Do not fallback automatically.
        if not self.active_client or not self.active_model:
            return f"AI Error: Selected LLM '{self.llm_manager.get_selected()}' is not configured or unavailable."

        try:
            logger.debug(f"Calling active LLM: {self.active_name}")
            return self._call_llm(self.active_client, self.active_model, prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Active LLM '{self.active_name}' failed: {e}")
            return f"AI Error: Selected LLM '{self.active_name}' is not working: {str(e)}"
    
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
            "active_llm": self.llm_manager.get_selected(),
            "active_model": self.active_model,
            "available_llms": self.llm_manager.get_names(),
        }
    
    def refresh_online_status(self) -> bool:
        """Refresh active LLM availability status"""
        ok = self._test_active_llm()
        logger.info(f"Active LLM status refreshed: {'✅' if ok else '❌'}")
        return ok

    def set_active_llm(self, name: str) -> str:
        """Select a different LLM by name (must be registered in LLMManager)."""
        if not self.llm_manager.set_selected(name):
            return f"❌ Unknown LLM: {name}"
        # Reinitialize active client
        self._init_active_client()
        ok = self._test_active_llm()
        if ok:
            return f"✅ Active LLM set to {name}"
        else:
            return f"⚠️ Active LLM set to {name}, but connection test failed"

    def get_mode_status(self) -> str:
        """Get current mode status as a string"""
        sel = self.llm_manager.get_selected()
        return f"Active LLM: {sel}"

    def get_llm_config(self) -> dict:
        """
        Provides LLM configuration for CrewAI based on availability.
        Note: LiteLLM is used internally by CrewAI and is working correctly.
        """
        # Provide the selected LLM config (for external libraries like CrewAI)
        sel = self.llm_manager.get_selected()
        cfg = self.llm_manager.get_config(sel) or {}
        logger.info(f"Providing LLM config for CrewAI: {sel}")
        # Return a sanitized config suitable for LLM wrappers
        return {
            "model": cfg.get("model"),
            "base_url": cfg.get("base_url"),
            "api_key": cfg.get("api_key"),
            "temperature": self.llm_temperature,
            "timeout": cfg.get("timeout", self.llm_timeout),
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

def set_active_llm(name: str) -> str:
    """Set the active LLM by name (registered in LLMManager)."""
    return ai_engine.set_active_llm(name)

def get_mode_status() -> str:
    """Get current mode status"""
    return ai_engine.get_mode_status()


