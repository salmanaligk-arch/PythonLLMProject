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
import json


class LLMManager:
    def __init__(self):
        # Store configs keyed by a friendly name
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._selected: Optional[str] = None
        self._storage_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(self._storage_dir, exist_ok=True)
        self._storage_file = os.path.join(self._storage_dir, "llm_models.json")
        # Ensure the storage file exists; if not create an empty payload
        if not os.path.exists(self._storage_file):
            try:
                with open(self._storage_file, 'w', encoding='utf-8') as f:
                    json.dump({"configs": {}, "selected": None}, f, indent=2)
            except Exception:
                logger.info(f"Could not create storage file: {self._storage_file}")
        # Try to load persisted configs (do NOT bootstrap from environment)
        self.load()

    def register(self, name: str, config: Dict[str, Any]):
        """Register a new LLM configuration.

        config should contain keys like: model, base_url, api_key, timeout, max_retries
        """
        self._configs[name] = dict(config)
        logger.info(f"Registered LLM: {name}")
        if not self._selected:
            self._selected = name
        # Persist changes
        try:
            self.save()
        except Exception:
            logger.info("Could not persist LLM configs")

    def get_names(self) -> List[str]:
        return list(self._configs.keys())

    def set_selected(self, name: str) -> bool:
        if name in self._configs:
            self._selected = name
            logger.info(f"Selected active LLM: {name}")
            try:
                self.save()
            except Exception:
                logger.info("Could not persist selected LLM")
            return True
        logger.warning(f"Attempted to select unknown LLM: {name}")
        return False

    def get_selected(self) -> Optional[str]:
        return self._selected

    def get_config(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if name is None:
            name = self._selected
        return self._configs.get(name)

    def save(self):
        payload = {
            "configs": self._configs,
            "selected": self._selected,
        }
        with open(self._storage_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def load(self):
        try:
            if os.path.exists(self._storage_file):
                with open(self._storage_file, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                self._configs = payload.get('configs', {}) or {}
                self._selected = payload.get('selected')
                logger.info(f"Loaded LLM configs from {self._storage_file}")
        except Exception as e:
            logger.warning(f"Failed to load LLM configs: {e}")

    def remove(self, name: str) -> bool:
        """Remove a registered LLM by name and persist changes."""
        if name in self._configs:
            del self._configs[name]
            if self._selected == name:
                # choose another selected if possible
                self._selected = next(iter(self._configs), None)
            try:
                self.save()
            except Exception:
                logger.info("Could not persist LLM configs after remove")
            logger.info(f"Removed LLM: {name}")
            return True
        return False

    def edit(self, name: str, updates: Dict[str, Any]) -> bool:
        """Edit an existing LLM configuration by merging `updates` into it and persisting."""
        if name not in self._configs:
            logger.warning(f"Attempted to edit unknown LLM: {name}")
            return False
        # Merge updates into existing config
        self._configs[name].update(dict(updates))
        try:
            self.save()
        except Exception:
            logger.info("Could not persist LLM configs after edit")
        logger.info(f"Edited LLM config: {name}")
        return True

    def test_model(self, cfg: Dict[str, Any], prompt: str = "hello") -> (bool, str):
        """Test an LLM configuration directly without registering it.

        cfg should contain keys: model, base_url, api_key, timeout, max_retries
        Returns (True, msg) on success or (False, error) on failure.
        """
        if not cfg:
            return False, "No config provided"
        model = cfg.get("model")
        base_url = cfg.get("base_url")

        if not base_url or not model:
            return False, "Missing base_url or model in config"
        # Use the AIEngine to create a client from the provided cfg and call the LLM
        # Import locally to avoid circular imports at module import time
        from services.chatbot import ai_engine

        client = ai_engine.create_client(cfg)
        if not client:
            return False, "Could not create client"

        try:
            # Reuse AIEngine's generic LLM call implementation
            resp_text = ai_engine.call_llm(client, model, prompt, max_tokens=8, temperature=float(cfg.get("temperature", 0.1)))
            from services.logger import logger
            logger.warning(f"LLM provider raw response: {resp_text}")
            if resp_text:
                return True, "Model responded"
            return False, f"No response from model. Raw response: {resp_text}"
        except Exception as e:
            from services.logger import logger
            logger.error(f"LLM test_model exception: {e}")
            return False, str(e)

    def get_llm_config(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Return a sanitized LLM configuration suitable for external LLM wrappers.

        If `name` is None the currently selected LLM is used.
        Returns a dict with keys: model, base_url, api_key, temperature, timeout.
        """
        cfg = self.get_config(name) or {}
        return {
            "model": cfg.get("model"),
            "base_url": cfg.get("base_url"),
            "api_key": cfg.get("api_key"),
            "temperature": cfg.get("temperature"),
            "timeout": cfg.get("timeout"),
        }


# Module-level manager: load persisted LLMs (or create defaults from env)
llm_manager = LLMManager()
