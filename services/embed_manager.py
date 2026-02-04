"""Embedding Manager: register and provide embedding models and functions"""
import os
import json
from typing import Dict, Any, Optional, List
from functools import lru_cache
import numpy as np
import requests
from services.logger import logger


class EmbedManager:
    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._selected: Optional[str] = None
        self._storage_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(self._storage_dir, exist_ok=True)
        self._storage_file = os.path.join(self._storage_dir, "embed_models.json")
        if not os.path.exists(self._storage_file):
            try:
                with open(self._storage_file, 'w', encoding='utf-8') as f:
                    json.dump({"configs": {}, "selected": None}, f, indent=2)
            except Exception:
                logger.info(f"Could not create embed storage file: {self._storage_file}")
        self.load()

    def register(self, name: str, config: Dict[str, Any]):
        self._configs[name] = dict(config)
        logger.info(f"Registered embedding model: {name}")
        if not self._selected:
            self._selected = name
        self.save()
        self.get_embedding.cache_clear()

    def get_names(self) -> List[str]:
        return list(self._configs.keys())

    def set_selected(self, name: str) -> bool:
        if name in self._configs:
            self._selected = name
            logger.info(f"Selected embedding model: {name}")
            self.save()
            return True
        logger.warning(f"Attempted to select unknown embedding model: {name}")
        return False

    def get_selected(self) -> Optional[str]:
        return self._selected

    def get_config(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        name = name or self._selected
        return self._configs.get(name)

    def _request_embedding(self, text: str, params: Dict[str, Any]) -> Optional[List[float]]:
        """Make a request to an embedding provider and return the embedding."""
        url = params.get("url")
        if not url:
            return None

        headers = {"Content-Type": "application/json"}
        if params.get("api_key"):
            headers["Authorization"] = f"Bearer {params.get('api_key')}"

        provider = params.get("provider", "generic")
        payload = {"model": params.get("model")}
        if provider == "openai":
            payload["input"] = text
        else: # generic/ollama
            payload["prompt"] = text
            payload["inputs"] = text

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=params.get("timeout", 30))
            r.raise_for_status()
            data = r.json()
            logger.info(f"Embedding call to {url} successful.")
            return self._parse_embedding_response(data)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Exception during embedding call to {url}: {e}")
            return None
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from {url}")
            return None

    def _parse_embedding_response(self, data: Any) -> Optional[List[float]]:
        """Parse common embedding response shapes into a plain list of floats."""
        if not data:
            return None
        if isinstance(data, list):
            return data[0] if isinstance(data[0], list) else data
        if isinstance(data, dict):
            if "embedding" in data:
                return data["embedding"][0] if isinstance(data["embedding"][0], list) else data["embedding"]
            if "embeddings" in data:
                return data["embeddings"][0] if isinstance(data["embeddings"][0], list) else data["embeddings"]
            if "data" in data and isinstance(data["data"], list) and data["data"]:
                return data["data"][0].get("embedding")
        return None

    @lru_cache(maxsize=128)
    def get_embedding(self, text: str, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Return embedding vector (numpy array) for the given text using selected model."""
        cfg = self.get_config(name)
        if not cfg:
            return None

        emb = self._request_embedding(text, cfg)

        if emb and isinstance(emb, list):
            try:
                return np.array(emb, dtype=np.float32)
            except (ValueError, TypeError):
                logger.info("Exception coercing embedding to numeric array")
                return None
        return None

    def save(self):
        try:
            with open(self._storage_file, 'w', encoding='utf-8') as f:
                json.dump({"configs": self._configs, "selected": self._selected}, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist embed configs: {e}")

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
            self.save()
            self.get_embedding.cache_clear()
            logger.info(f"Removed embed model: {name}")
            return True
        return False

    def test_model(self, name: Optional[str] = None, text: str = "test", params: Optional[Dict[str, Any]] = None) -> (bool, str):
        """Perform a quick smoke-test of the named embedding model."""
        if not params or not isinstance(params, dict):
            return False, "No parameters provided for test"

        emb = self._request_embedding(text, params)

        if emb and isinstance(emb, list):
            try:
                arr = np.array(emb, dtype=np.float32)
                return True, str(arr.shape)
            except (ValueError, TypeError):
                return False, "Failed to coerce embedding to numeric array"
        
        logger.warning(f"Embedding provider raw response for test: {emb}")
        return False, "Unrecognized or empty embedding response"


# Create module-level embed manager and register defaults from env
embed_manager = EmbedManager()