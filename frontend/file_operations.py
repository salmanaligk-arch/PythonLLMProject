"""
File Operations for the UI
Handles document upload, file management, and file browser functionality
"""

import gradio as gr
from services.vector_store import vector_store
from services.logger import logger
from services.embed_manager import embed_manager
import os
import datetime
import io

def _read_uploaded_file(file):
    """
    Robustly read an uploaded file object and return its content and filename.
    Supports various Gradio file object types.
    """
    # fobj may have .path (a filesystem path)
    if hasattr(file, 'path') and getattr(file, 'path') and os.path.exists(getattr(file, 'path')):
        p = getattr(file, 'path')
        with open(p, 'rb') as fh:
            return fh.read(), os.path.basename(getattr(file, 'name', p))
    # fobj may have .name that is a real path
    if hasattr(file, 'name') and os.path.exists(getattr(file, 'name')):
        p = getattr(file, 'name')
        with open(p, 'rb') as fh:
            return fh.read(), os.path.basename(p)
    # fobj may expose a file-like attribute
    if hasattr(file, 'file'):
        fh = getattr(file, 'file')
        try:
            fh.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        data = fh.read()
        if isinstance(data, str):
            data = data.encode('utf-8')
        return data, os.path.basename(getattr(file, 'name', 'uploaded'))
    # fobj may be a dict from some uploaders
    if isinstance(file, dict):
        data = file.get('data') or file.get('content')
        name = file.get('name', 'uploaded')
        if isinstance(data, str):
            data = data.encode('utf-8')
        return data, name
    # raw bytes
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), 'uploaded'
    # file-like object
    if hasattr(file, 'read'):
        try:
            data = file.read()
            if isinstance(data, str):
                data = data.encode('utf-8')
            return data, os.path.basename(getattr(file, 'name', 'uploaded'))
        except Exception:
            pass
    raise ValueError('Unsupported uploaded file object')

def upload_documents(files, embed_model=None, chunk_size: int = 1000, overlap: int = 200):
    """Upload and index multiple documents"""
    if not files:
        return "No files were uploaded. Please select files to upload."
    
    if embed_model:
        try:
            embed_manager.set_selected(embed_model)
        except Exception as e:
            logger.error(f"Failed to set embedding model: {e}")
            return f"Error: Failed to set embedding model '{embed_model}'. Please check settings."

    results = []
    
    for file in files:
        filename = getattr(file, 'name', 'unknown file')
        try:
            content, filename = _read_uploaded_file(file)

            if not any(filename.endswith(ext) for ext in ['.pdf', '.txt', '.docx', '.xlsx']):
                results.append(f"✗ Unsupported file type: {filename}")
                logger.warning(f"Skipped unsupported file type: {filename}")
                continue
            
            vector_store.add_file(content, filename, chunk_size=chunk_size, overlap=overlap)
            results.append(f"✓ Successfully uploaded and indexed: {filename}")
            logger.info(f"Successfully indexed '{filename}' with chunk_size={chunk_size}, overlap={overlap}")
        
        except Exception as e:
            error_message = f"✗ Error processing {filename}: {str(e)}"
            results.append(error_message)
            logger.error(f"Failed to upload or index {filename}: {e}", exc_info=True)
    
    try:
        vector_store.save_index()
        logger.info("FAISS index saved successfully after document upload.")
    except Exception as e:
        save_error = "Critical: Failed to save the vector index. Changes may be lost."
        results.append(save_error)
        logger.critical(f"{save_error} Error: {e}", exc_info=True)
    
    return "\n".join(results)

def get_files_list():
    """Get list of all uploaded files for the file browser"""
    files = vector_store.get_all_files()
    if not files:
        return []
    
    file_data = []
    for file_info in files:
        filename = file_info.get('filename', 'Unknown')
        file_type = file_info.get('file_type', 'N/A')
        file_size = file_info.get('file_size', 0)
        word_count = file_info.get('total_word_count', 0)
        timestamp_str = file_info.get('upload_timestamp', '')
        
        formatted_timestamp = 'N/A'
        if timestamp_str:
            try:
                # Attempt to parse ISO 8601 format, handling 'Z'
                dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                # Fallback for other common formats or partial data
                formatted_timestamp = timestamp_str.split('.')[0].replace('T', ' ')

        file_data.append([
            filename,
            file_type.upper(),
            f"{file_size/1024:.2f} KB" if file_size > 1024 else f"{file_size} B",
            f"{word_count:,}",
            formatted_timestamp
        ])
    return file_data

def refresh_files():
    """Refresh the files list in the UI"""
    return get_files_list()

def select_file(evt: gr.SelectData):
    """Handle file selection from DataFrame click"""
    if not evt or evt.index is None:
        return "Click on a file row to select it.", None, "No file selected"
            
    try:
        row_index = evt.index[0]
        files = vector_store.get_all_files()
        
        if row_index >= len(files):
            logger.warning(f"Selection index {row_index} is out of range for files list length {len(files)}.")
            return "Selected row is out of range. Please refresh the file list.", None, "Selection error"
            
        selected_file_info = files[row_index]
        filename = selected_file_info['filename']
        logger.info(f"UI selected file: '{filename}' at index {row_index}")
        
        content = vector_store.get_file_content(filename)
        
        if content is None:
            logger.warning(f"No content found for file: {filename}")
            return f"No content found for {filename}. The file might be empty or corrupted.", filename, f"Selected: {filename} (no content)"
            
        return content, filename, f"Selected: {filename}"
        
    except IndexError:
        logger.warning("IndexError in select_file, possibly due to an empty or outdated file list.")
        return "Please select a file from the list. You may need to refresh.", None, "No file selected"
    except Exception as e:
        logger.error(f"An unexpected error occurred in select_file: {e}", exc_info=True)
        return f"An unexpected error occurred while selecting the file: {e}", None, "Error"

def delete_selected_file(filename):
    """Delete a selected file from the vector store"""
    if not filename or not filename.strip():
        return "⚠️ No file selected. Click a file in the table to select it, then click Delete.", get_files_list(), ""
    
    logger.info(f"Delete operation initiated for filename: '{filename}'")
    
    try:
        if vector_store.delete_file(filename):
            vector_store.save_index()
            logger.info(f"Successfully deleted file: {filename} and saved index.")
            return f"✅ Successfully deleted: {filename}", get_files_list(), "File deleted. Select another file to view."
        else:
            logger.warning(f"File '{filename}' not found for deletion. It may have been already deleted.")
            return f"❌ File not found: '{filename}'. It may have been deleted already. Refreshing list.", get_files_list(), ""
    except Exception as e:
        logger.error(f"Error during file deletion for '{filename}': {e}", exc_info=True)
        return f"❌ An error occurred while deleting the file: {e}", get_files_list(), ""