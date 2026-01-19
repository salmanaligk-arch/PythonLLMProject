import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from .base_handler import FileHandler, chunk_text

class ExcelHandler(FileHandler):
    def process(self, content: bytes, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        df = pd.read_excel(content)
        text_content = df.to_string()
        chunks = chunk_text(text_content, chunk_size=chunk_size, overlap=overlap)
        excel_metadata = self.get_metadata(content, filename)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **excel_metadata,
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_text_length': len(chunk),
                'chunk_word_count': len(chunk.split())
            }
            processed_chunks.append({'content': chunk, 'metadata': chunk_metadata})
        return processed_chunks

    def get_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        df = pd.read_excel(content)
        return {
            'filename': filename,
            'file_type': 'excel',
            'file_size': len(content),
            'upload_timestamp': datetime.now().isoformat(),
            'rows': len(df),
            'columns': len(df.columns),
        }
