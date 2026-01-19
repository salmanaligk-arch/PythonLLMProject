from datetime import datetime
from typing import List, Dict, Any
from .base_handler import FileHandler, chunk_text

class TXTHandler(FileHandler):
    def process(self, content: bytes, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        text_content = content.decode('utf-8')
        chunks = chunk_text(text_content, chunk_size=chunk_size, overlap=overlap)
        text_metadata = self.get_metadata(content, filename)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **text_metadata,
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_text_length': len(chunk),
                'chunk_word_count': len(chunk.split())
            }
            processed_chunks.append({'content': chunk, 'metadata': chunk_metadata})
        return processed_chunks

    def get_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        text_content = content.decode('utf-8')
        return {
            'filename': filename,
            'file_type': 'text',
            'file_size': len(content),
            'upload_timestamp': datetime.now().isoformat(),
            'total_text_length': len(text_content),
            'total_word_count': len(text_content.split()),
            'line_count': len(text_content.split('\n'))
        }
