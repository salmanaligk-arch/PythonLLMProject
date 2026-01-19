import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
import io
import os
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any
from .base_handler import FileHandler, chunk_text
from services.logger import logger

# Set the path to the Tesseract executable for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PDFHandler(FileHandler):
    def process(self, content: bytes, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        full_text = self._extract_text_from_pdf(content)
        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        pdf_metadata = self.get_metadata(content, filename)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **pdf_metadata,
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_text_length': len(chunk),
                'chunk_word_count': len(chunk.split())
            }
            processed_chunks.append({'content': chunk, 'metadata': chunk_metadata})
        return processed_chunks

    def get_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                return {
                    'filename': filename,
                    'file_type': 'pdf',
                    'file_size': len(content),
                    'upload_timestamp': datetime.now().isoformat(),
                    'total_pages': len(pdf.pages),
                    'pdf_title': pdf.metadata.get('Title', ''),
                    'pdf_author': pdf.metadata.get('Author', ''),
                }
        except Exception as e:
            logger.error(f"Error getting metadata with pdfplumber for {filename}: {e}")
            return {
                'filename': filename,
                'file_type': 'pdf',
                'file_size': len(content),
                'upload_timestamp': datetime.now().isoformat(),
            }

    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        full_text = ""
        try:
            with pdfplumber.open(BytesIO(pdf_content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"Processing page {page_num + 1} with pdfplumber")
                    
                    text = page.extract_text()
                    
                    if text and text.strip():
                        logger.info("Extracting text and tables from text-based page.")
                        tables = page.extract_tables()
                        for table in tables:
                            table_text = "\n".join(["\t".join([cell if cell is not None else "" for cell in row]) for row in table if row])
                            text += "\n" + table_text
                    else:
                        logger.info("Page seems to be image-based, using OCR.")
                        img = page.to_image(resolution=300).original
                        img = img.convert("L")
                        text = pytesseract.image_to_string(img, lang="eng")
                    
                    text = re.sub(r'\n\s*\n+', '    ', text)
                    full_text += text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed with error: {e}. Falling back to PyMuPDF.")
            try:
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                for page_num in range(len(doc)):
                    logger.info(f"Processing page {page_num + 1} with PyMuPDF")
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if not text.strip():
                        logger.info("PyMuPDF found no text, using OCR as fallback.")
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        img = img.convert("L")
                        text = pytesseract.image_to_string(img, lang="eng")
                    
                    text = re.sub(r'\n\s*\n+', '    ', text)
                    full_text += text + "\n"
            except Exception as fitz_e:
                logger.error(f"PyMuPDF fallback also failed: {fitz_e}")

        return full_text.strip()
