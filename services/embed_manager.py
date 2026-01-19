"""Embedding Manager: register and provide embedding models and functions"""
import os
import json
from typing import Dict, Any, Optional, List
import numpy as np
import ollama
from services.logger import logger


class EmbedManager:
    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._selected: Optional[str] = None
        self._storage_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(self._storage_dir, exist_ok=True)
        self._storage_file = os.path.join(self._storage_dir, "embed_models.json")
        # Ensure storage file exists; do not create defaults from env
        if not os.path.exists(self._storage_file):
            try:
                with open(self._storage_file, 'w', encoding='utf-8') as f:
                    json.dump({"configs": {}, "selected": None}, f, indent=2)
            except Exception:
                logger.debug(f"Could not create embed storage file: {self._storage_file}")
        self.load()

    def register(self, name: str, config: Dict[str, Any]):
        self._configs[name] = dict(config)
        logger.info(f"Registered embedding model: {name}")
        if not self._selected:
            self._selected = name
        try:
            self.save()
        except Exception:
            logger.debug("Could not persist embed configs")

    def get_names(self) -> List[str]:
        return list(self._configs.keys())

    def set_selected(self, name: str) -> bool:
        if name in self._configs:
            self._selected = name
            logger.info(f"Selected embedding model: {name}")
            try:
                self.save()
            except Exception:
                logger.debug("Could not persist selected embed model")
            return True
        logger.warning(f"Attempted to select unknown embedding model: {name}")
        return False

    def get_selected(self) -> Optional[str]:
        return self._selected

    def get_config(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if name is None:
            name = self._selected
        return self._configs.get(name)

    def get_embedding(self, text: str, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Return embedding vector (numpy array) for the given text using selected model."""
        cfg = self.get_config(name)
        if not cfg:
            logger.warning(f"No embedding config for: {name}")
            return None

        provider = cfg.get("provider", "ollama")
        model = cfg.get("model")

        try:
            if provider == "ollama":
                resp = ollama.embeddings(model=model, prompt=text)
                emb = np.array(resp.get('embedding', []), dtype=np.float32)
                return emb
            else:
                # Default to ollama if unknown provider for now
                logger.warning(f"Unknown embedding provider '{provider}', defaulting to ollama")
                resp = ollama.embeddings(model=model, prompt=text)
                emb = np.array(resp.get('embedding', []), dtype=np.float32)
                return emb
        except Exception as e:
            logger.error(f"Embedding generation failed for model {name}: {e}")
            return None

    def save(self):
        payload = {"configs": self._configs, "selected": self._selected}
        with open(self._storage_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def load(self):
        try:
            if os.path.exists(self._storage_file):
                with open(self._storage_file, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                self._configs = payload.get('configs', {}) or {}
                self._selected = payload.get('selected')
                logger.info(f"Loaded embed configs from {self._storage_file}")
        except Exception as e:
            logger.warning(f"Failed to load embed configs: {e}")

    def remove(self, name: str) -> bool:
        if name in self._configs:
            del self._configs[name]
            if self._selected == name:
                self._selected = next(iter(self._configs), None)
            try:
                self.save()
            except Exception:
                logger.debug("Could not persist embed configs after remove")
            logger.info(f"Removed embed model: {name}")
            return True
        return False


# Create module-level embed manager and register defaults from env
embed_manager = EmbedManager()
"""
default_embed = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:v1.5")
embed_manager.register(
    "default",
    {
        "provider": "ollama",
        "model": default_embed,
        "timeout": int(os.getenv("EMBEDDING_TIMEOUT", 30)),
    },
)
"""