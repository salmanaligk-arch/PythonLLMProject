"""Embedding Manager: register and provide embedding models and functions"""
import os
import json
from typing import Dict, Any, Optional, List, Tuple
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
        # Ensure storage file exists; do not create defaults from env
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
        try:
            self.save()
        except Exception:
            logger.info("Could not persist embed configs")

    def get_names(self) -> List[str]:
        return list(self._configs.keys())

    def set_selected(self, name: str) -> bool:
        if name in self._configs:
            self._selected = name
            logger.info(f"Selected embedding model: {name}")
            try:
                self.save()
            except Exception:
                logger.info("Could not persist selected embed model")
            return True
        logger.warning(f"Attempted to select unknown embedding model: {name}")
        return False

    def get_selected(self) -> Optional[str]:
        return self._selected

    def get_config(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if name is None:
            name = self._selected
        return self._configs.get(name)

    def call_embedding(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 30):
        """Perform the HTTP POST to the embedding provider and return parsed JSON or None."""
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            logger.info(f"Embedding call to {url} returned status {r.json()}")
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            logger.warning(f"Exception during embedding call to {url}: {e}")
            return None

    def _parse_embedding_response(self, data: Any) -> Optional[List[float]]:
        """Parse common embedding response shapes into a plain list of floats.

        Supported shapes (non-exhaustive):
        - {'embedding': [...]}
        - {'embeddings': [[...], ...]}
        - {'data': [{'embedding': [...]}, ...]}
        - [[...], ...] or [{'embedding': [...]}, ...]
        Returns the first embedding found as a list or None.
        """
        if not data:
            return None

        if isinstance(data, dict):
            emb = data.get("embedding") or data.get("embeddings")
            if emb and isinstance(emb, list):
                # embeddings may be nested lists ([ [..] ])
                if emb and isinstance(emb[0], list):
                    return emb[0]
                return emb
            if isinstance(data.get("data"), list) and data["data"]:
                first = data["data"][0]
                if isinstance(first, dict):
                    return first.get("embedding")

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, list):
                return first
            if isinstance(first, dict):
                return first.get("embedding")

        return None

    def get_embedding(self, text: str, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Return embedding vector (numpy array) for the given text using selected model.

        This is a compact, minimal implementation: one request and a few common
        response-shape checks. On any error or unknown shape it returns None.
        """
        cfg = self.get_config(name)
        if not cfg:
            return None

        url = cfg.get("url")
        if not url:
            return None

        headers = {}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg.get('api_key')}"

        payload = {"model": cfg.get("model"), "prompt": text, "input": text}

        data = self.call_embedding(url, headers, payload, timeout=cfg.get("timeout", 30))
        if data is None:
            return None

        emb = self._parse_embedding_response(data)
        if emb and isinstance(emb, list):
            try:
                return np.array(emb, dtype=np.float32)
            except Exception:
                return None
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
                logger.info("Could not persist embed configs after remove")
            logger.info(f"Removed embed model: {name}")
            return True
        return False

    def edit(self, name: str, updates: Dict[str, Any]) -> bool:
        """Edit an existing embedding model configuration and persist changes."""
        if name not in self._configs:
            logger.warning(f"Attempted to edit unknown embed model: {name}")
            return False
        self._configs[name].update(dict(updates))
        try:
            self.save()
        except Exception:
            logger.info("Could not persist embed configs after edit")
        logger.info(f"Edited embed model config: {name}")
        return True

    def test_model(self, name: Optional[str] = None, text: str = "test", params: Optional[Dict[str, Any]] = None) -> (bool, str):
        """Perform a quick smoke-test of the named embedding model.

        Returns (True, dims) on success (dims as string) or (False, error_message) on failure.
        """
        # Require params from UI; do not read request params from stored config
        if not params or not isinstance(params, dict):
            return False, "No parameters provided for test"

        # Build headers from params (e.g. api_key)
        headers: Dict[str, str] = {}
        if params.get("api_key"):
            headers["Authorization"] = f"Bearer {params.get('api_key')}"
        headers["Content-Type"]= "application/json"
        # Build payload from params (UI controls). Exclude control-only keys.
        # payload = {k: v for k, v in params.items() if k not in ("url", "api_key", "timeout")}
        # payload["input"] = text
        payload = {"model": params.get("model"), "prompt": text, "input": text}
        url = params.get("url")
        timeout = params.get("timeout", 30)
        if not url:
            return False, "No URL provided in parameters"

        data = self.call_embedding(url, headers, payload, timeout=timeout)
        if not data:
            return False, "No response from embedding provider"

        # Log the raw response for debugging
        logger.warning(f"Embedding provider raw response: {data}")
        emb = self._parse_embedding_response(data)
        if emb and isinstance(emb, list):
            try:
                arr = np.array(emb, dtype=np.float32)
                return True, str(arr.shape)
            except Exception:
                return False, "Failed to coerce embedding to numeric array"
        return False, f"Unrecognized embedding response shape: {type(data)} {data}"


# Create module-level embed manager and register defaults from env
embed_manager = EmbedManager()