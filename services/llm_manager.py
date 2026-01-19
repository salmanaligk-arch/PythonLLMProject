"""LLM Manager: register and create LLM clients from configuration

This module provides a general `LLMManager` class that holds multiple
LLM definitions and can instantiate client objects on demand. It does
not hardcode specific providers inside the class; instead configs are
registered from the environment (or any other external source).
"""
import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from services.logger import logger


class LLMManager:
    def __init__(self):
        # Store configs keyed by a friendly name
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._selected: Optional[str] = None

    def register(self, name: str, config: Dict[str, Any]):
        """Register a new LLM configuration.

        config should contain keys like: model, base_url, api_key, timeout, max_retries
        """
        self._configs[name] = dict(config)
        logger.info(f"Registered LLM: {name}")
        if not self._selected:
            self._selected = name

    def get_names(self) -> List[str]:
        return list(self._configs.keys())

    def set_selected(self, name: str) -> bool:
        if name in self._configs:
            self._selected = name
            logger.info(f"Selected active LLM: {name}")
            return True
        logger.warning(f"Attempted to select unknown LLM: {name}")
        return False

    def get_selected(self) -> Optional[str]:
        return self._selected

    def get_config(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if name is None:
            name = self._selected
        return self._configs.get(name)

    def create_client(self, name: Optional[str] = None) -> Optional[OpenAI]:
        """Create and return an OpenAI-compatible client for the named LLM.

        Returns None if configuration is missing.
        """
        cfg = self.get_config(name)
        if not cfg:
            logger.warning(f"No LLM config found for: {name}")
            return None

        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")
        timeout = int(cfg.get("timeout", 30))
        max_retries = int(cfg.get("max_retries", 2))

        try:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
            return client
        except Exception as e:
            logger.error(f"Error creating client for {name}: {e}")
            return None


# Module-level manager: load basic LLMs from environment
llm_manager = LLMManager()

# Register HuggingFace online LLM if env present
hf_token = os.getenv("HF_TOKEN")
hf_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
if hf_token and hf_token != "your_huggingface_token_here":
    llm_manager.register(
        "huggingface",
        {
            "provider": "huggingface",
            "model": hf_model,
            "base_url": "https://router.huggingface.co/v1",
            "api_key": hf_token,
            "timeout": int(os.getenv("LLM_TIMEOUT", 30)),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", 2)),
        },
    )

# Register Ollama offline LLM (local) if present
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
llm_manager.register(
    "ollama",
    {
        "provider": "ollama",
        "model": ollama_model,
        "base_url": f"{ollama_url}/v1",
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        "timeout": max(1800, int(os.getenv("LLM_TIMEOUT", 30))),
        "max_retries": 1,
    },
)
