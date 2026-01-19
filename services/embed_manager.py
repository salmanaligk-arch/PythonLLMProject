"""Embedding Manager: register and provide embedding models and functions"""
import os
from typing import Dict, Any, Optional, List
import numpy as np
import ollama
from services.logger import logger


class EmbedManager:
    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._selected: Optional[str] = None

    def register(self, name: str, config: Dict[str, Any]):
        self._configs[name] = dict(config)
        logger.info(f"Registered embedding model: {name}")
        if not self._selected:
            self._selected = name

    def get_names(self) -> List[str]:
        return list(self._configs.keys())

    def set_selected(self, name: str) -> bool:
        if name in self._configs:
            self._selected = name
            logger.info(f"Selected embedding model: {name}")
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


# Create module-level embed manager and register defaults from env
embed_manager = EmbedManager()

default_embed = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:v1.5")
embed_manager.register(
    "default",
    {
        "provider": "ollama",
        "model": default_embed,
        "timeout": int(os.getenv("EMBEDDING_TIMEOUT", 30)),
    },
)
