import os
from typing import Dict, Any, List, Optional

def handle_temp_document(file_content: bytes, filename: str) -> str:
    """Handle temporary document content extraction for all supported file types"""
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.pdf':
        from services.file_handlers.pdf_handler import PDFHandler
        handler = PDFHandler()
        return handler._extract_text_from_pdf(file_content)
    elif file_ext == '.txt':
        from services.file_handlers.txt_handler import TXTHandler
        handler = TXTHandler()
        return file_content.decode('utf-8')
    elif file_ext == '.docx':
        from services.file_handlers.docx_handler import DocxHandler
        handler = DocxHandler()
        processed_chunks = handler.process(file_content, filename)
        return '\n\n'.join([chunk['content'] for chunk in processed_chunks])
    elif file_ext == '.xlsx':
        from services.file_handlers.excel_handler import ExcelHandler
        handler = ExcelHandler()
        processed_chunks = handler.process(file_content, filename)
        return '\n\n'.join([chunk['content'] for chunk in processed_chunks])
    else:
        # Fallback to text for unknown types
        return file_content.decode('utf-8', errors='ignore')

def process_temp_document(temp_document_file) -> Optional[str]:
    """Process temporary document file - handles availability check, reading, and content extraction"""
    if temp_document_file is None:
        return None
        
    try:
        with open(temp_document_file.name, 'rb') as f:
            content = f.read()
        
        filename = os.path.basename(temp_document_file.name)
        return handle_temp_document(content, filename)
        
    except Exception as e:
        raise Exception(f"Error reading temporary document: {str(e)}")