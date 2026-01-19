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
                logger.debug(f"Could not create storage file: {self._storage_file}")
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
            logger.debug("Could not persist LLM configs")

    def get_names(self) -> List[str]:
        return list(self._configs.keys())

    def set_selected(self, name: str) -> bool:
        if name in self._configs:
            self._selected = name
            logger.info(f"Selected active LLM: {name}")
            try:
                self.save()
            except Exception:
                logger.debug("Could not persist selected LLM")
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
                logger.debug("Could not persist LLM configs after remove")
            logger.info(f"Removed LLM: {name}")
            return True
        return False


# Module-level manager: load persisted LLMs (or create defaults from env)
llm_manager = LLMManager()
