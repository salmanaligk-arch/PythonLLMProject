import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
import ollama
from typing import List, Dict, Any, Optional
from datetime import datetime
from services.file_handlers.pdf_handler import PDFHandler
from services.file_handlers.txt_handler import TXTHandler
from services.file_handlers.excel_handler import ExcelHandler
from services.file_handlers.docx_handler import DocxHandler
from services.logger import logger
from services.embed_manager import embed_manager

# Load environment variables
load_dotenv()
load_dotenv("config.env")

class FAISSVectorStore:
    def __init__(self, dimension: int = None, index_path: str = None):
        # Use environment variables for configuration
        self.dimension = dimension or int(os.getenv("VECTOR_DIMENSION", 768))
        self.index_path = index_path or os.getenv("VECTOR_INDEX_PATH", "faiss_indexes")
        os.makedirs(self.index_path, exist_ok=True)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
        self.file_handlers = {
            '.pdf': PDFHandler(),
            '.txt': TXTHandler(),
            '.xlsx': ExcelHandler(),
            '.docx': DocxHandler(),
        }
        
    def get_embedding(self, text: str) -> np.ndarray:
        # Use embed_manager to obtain embedding for the selected model
        emb = embed_manager.get_embedding(text)
        if emb is None:
            raise RuntimeError("Embedding generation failed")
        return emb
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        embedding = self.get_embedding(content)
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        self.index.add(embedding)
        self.documents.append(content)
        
        # Add enhanced metadata with content analysis
        enhanced_metadata = metadata or {}
        enhanced_metadata.update({
            'content_length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len([s for s in content.split('.') if s.strip()]),
            'added_timestamp': datetime.now().isoformat()
        })
        
        self.metadata.append(enhanced_metadata)

    def add_file(self, file_content: bytes, filename: str, chunk_size: int = 1000, overlap: int = 200, embed_model: Optional[str] = None):
        file_ext = os.path.splitext(filename)[1].lower()
        handler = self.file_handlers.get(file_ext)
        
        if not handler:
            logger.error(f"Unsupported file type: {file_ext}")
            return

        # If requested, switch embedding model for this upload
        if embed_model:
            try:
                embed_manager.set_selected(embed_model)
            except Exception:
                logger.warning(f"Could not set embed model to {embed_model}")

        processed_chunks = handler.process(file_content, filename, chunk_size=chunk_size, overlap=overlap)
        for chunk in processed_chunks:
            self.add_document(chunk['content'], chunk['metadata'])
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        # Check if index has any documents
        if self.index.ntotal == 0 or len(self.documents) == 0:
            return []
        
        # Use environment variable for search parameter if not provided
        k = k or int(os.getenv("FAISS_SEARCH_K", 5))
        
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Limit k to available documents
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            # FAISS returns -1 for missing results, so check idx >= 0
            if 0 <= idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(scores[0][i])
                })
        return results
    
    def save_index(self, filename: str = "vector_index"):
        index_file = os.path.join(self.index_path, f"{filename}.index")
        data_file = os.path.join(self.index_path, f"{filename}.pkl")
        
        faiss.write_index(self.index, index_file)
        
        with open(data_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    def load_index(self, filename: str = "vector_index"):
        index_file = os.path.join(self.index_path, f"{filename}.index")
        data_file = os.path.join(self.index_path, f"{filename}.pkl")
        
        if os.path.exists(index_file) and os.path.exists(data_file):
            self.index = faiss.read_index(index_file)
            
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            return True
        return False
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        files = {}
        for meta in self.metadata:
            filename = meta.get('filename')
            if filename and filename not in files:
                files[filename] = {
                    'filename': filename,
                    'file_type': meta.get('file_type', 'unknown'),
                    'file_size': meta.get('file_size', 0),
                    'upload_timestamp': meta.get('upload_timestamp', ''),
                    'total_chunks': meta.get('total_chunks', 1),
                    'total_pages': meta.get('total_pages', 0),
                    'total_word_count': meta.get('total_word_count', 0)
                }
        return list(files.values())
    
    def get_file_content(self, filename: str) -> str:
        content_chunks = []
        for i, meta in enumerate(self.metadata):
            if meta.get('filename') == filename:
                content_chunks.append({
                    'chunk_id': meta.get('chunk_id', 0),
                    'content': self.documents[i]
                })
        
        # Sort by chunk_id and combine
        content_chunks.sort(key=lambda x: x['chunk_id'])
        return '\n\n'.join([chunk['content'] for chunk in content_chunks])
    
    def delete_file(self, filename: str) -> bool:
        """Delete all chunks associated with a file from the FAISS index"""
        try:
            # Find all indices to delete (in reverse order to avoid index shifting)
            indices_to_delete = []
            for i, meta in enumerate(self.metadata):
                if meta.get('filename') == filename:
                    indices_to_delete.append(i)
            
            if not indices_to_delete:
                return False
            
            # Delete in reverse order to maintain correct indices
            for idx in sorted(indices_to_delete, reverse=True):
                del self.documents[idx]
                del self.metadata[idx]
            
            # Rebuild FAISS index with remaining documents
            self.index = faiss.IndexFlatIP(self.dimension)
            for doc in self.documents:
                embedding = self.get_embedding(doc)
                embedding = embedding.reshape(1, -1)
                faiss.normalize_L2(embedding)
                self.index.add(embedding)
            
            return True
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
            return False

# Global vector store instance
vector_store = FAISSVectorStore()
vector_store.load_index()