from abc import ABC, abstractmethod
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

class FileHandler(ABC):
    @abstractmethod
    def process(self, content: bytes, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Process the file content and return a list of chunks with metadata."""
        pass

    @abstractmethod
    def get_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract metadata from the file."""
        pass
