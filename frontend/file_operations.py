"""
File Operations for the UI
Handles document upload, file management, and file browser functionality
"""

import gradio as gr
from services.vector_store import vector_store
from services.logger import logger
import os
def upload_documents(files, embed_model=None, chunk_size: int = 1000, overlap: int = 200):
    """Upload and index multiple documents"""
    if not files:
        return "No files uploaded"
    
    results = []
    
    for file in files:
        try:
            # Basic file validation
            # Robustly read uploaded file: support objects with .path, .name, .file,
            # Gradio NamedString/dicts, bytes, and file-like objects.
            def _read_uploaded(fobj):
                # fobj may have .path (a filesystem path)
                if hasattr(fobj, 'path') and getattr(fobj, 'path'):
                    p = getattr(fobj, 'path')
                    if os.path.exists(p):
                        with open(p, 'rb') as fh:
                            return fh.read(), os.path.basename(getattr(fobj, 'name', p))
                # fobj may have .name that is a real path
                if hasattr(fobj, 'name') and os.path.exists(getattr(fobj, 'name')):
                    p = getattr(fobj, 'name')
                    with open(p, 'rb') as fh:
                        return fh.read(), os.path.basename(p)
                # fobj may expose a file-like attribute
                if hasattr(fobj, 'file'):
                    fh = getattr(fobj, 'file')
                    try:
                        fh.seek(0)
                    except Exception:
                        pass
                    data = fh.read()
                    if isinstance(data, str):
                        data = data.encode('utf-8')
                    return data, os.path.basename(getattr(fobj, 'name', 'uploaded'))
                # fobj may be a dict from some uploaders
                if isinstance(fobj, dict):
                    data = fobj.get('data') or fobj.get('content')
                    name = fobj.get('name', 'uploaded')
                    if isinstance(data, str):
                        data = data.encode('utf-8')
                    return data, name
                # raw bytes
                if isinstance(fobj, (bytes, bytearray)):
                    return bytes(fobj), 'uploaded'
                # file-like object
                if hasattr(fobj, 'read'):
                    try:
                        data = fobj.read()
                        if isinstance(data, str):
                            data = data.encode('utf-8')
                        return data, os.path.basename(getattr(fobj, 'name', 'uploaded'))
                    except Exception:
                        pass
                raise ValueError('Unsupported uploaded file object')

            try:
                content, filename = _read_uploaded(file)
            except Exception as e:
                results.append(f"✗ Error uploading {getattr(file, 'name', str(file))}: {e}")
                continue

            if not filename.endswith(('.pdf', '.txt', '.docx', '.xlsx')):
                results.append(f"✗ Unsupported file type: {filename}")
                continue
            
            # Add to vector store with chunking and embedding selection
            vector_store.add_file(content, filename, chunk_size=chunk_size, overlap=overlap, embed_model=embed_model)
            
            results.append(f"✓ Successfully uploaded and indexed: {filename}")
        
        except Exception as e:
            results.append(f"✗ Error uploading {os.path.basename(file.name)}: {str(e)}")
    
    # Save the updated index
    vector_store.save_index()
    
    return "\n".join(results)

def get_files_list():
    """Get list of all uploaded files for the file browser"""
    files = vector_store.get_all_files()
    if not files:
        return []
    
    file_data = []
    for file_info in files:
        # Handle missing or None values gracefully
        filename = file_info.get('filename', 'Unknown')
        file_type = file_info.get('file_type', 'unknown')
        file_size = file_info.get('file_size', 0)
        word_count = file_info.get('total_word_count', 0)
        timestamp = file_info.get('upload_timestamp', '')
        
        # Format timestamp
        formatted_timestamp = timestamp[:19].replace('T', ' ') if timestamp else 'Unknown'
        
        file_data.append([
            filename,
            file_type.upper(),
            f"{file_size:,} bytes",
            f"{word_count:,} words",
            formatted_timestamp
        ])
    return file_data

def refresh_files():
    """Refresh the files list"""
    return get_files_list()

def select_file_by_name(filename):
    """Select and display file content by filename"""
    from services.logger import logger
    
    if not filename:
        return "Please select a file from the dropdown", None
    
    try:
        logger.info(f"Selecting file by name: {filename}")
        content = vector_store.get_file_content(filename)
        return content, filename
    except Exception as e:
        logger.error(f"Error selecting file {filename}: {e}")
        return f"Error loading file: {str(e)}", None

def select_file(evt: gr.SelectData):
    """Handle file selection from DataFrame click"""
    from services.logger import logger
    
    try:
        # Check if we have a proper selection event
        if not evt or not hasattr(evt, 'index'):
            return "Click on a file row to select it", None, "No file selected"
            
        # Get the row index from the selection event
        row_index = evt.index[0]
        logger.info(f"Selected row index: {row_index}")
        
        # Get all files from vector store
        files = vector_store.get_all_files()
        
        if row_index >= len(files):
            return "Selected row is out of range", None, "Selection error"
            
        # Get the filename from the selected row
        selected_file = files[row_index]
        filename = selected_file['filename']
        logger.info(f"Selected file: {filename}")
        
        # Get file content from vector store
        content = vector_store.get_file_content(filename)
        
        if not content:
            return f"No content found for {filename}", filename, f"Selected: {filename} (no content)"
            
        return content, filename, f"Selected: {filename}"
        
    except IndexError:
        logger.warning("IndexError in select_file, possibly no file selected or empty file list.")
        return "Please select a file from the list.", None, "No file selected"
    except Exception as e:
        logger.error(f"Error in select_file: {str(e)}")
        return f"Error selecting file: {str(e)}", None, "Error selecting file"

def delete_selected_file(filename):
    """Delete a selected file from the vector store"""
    from services.logger import logger
    
    logger.info(f"Delete requested for filename: '{filename}'")
    
    if not filename or filename.strip() == "":
        return "⚠️ No file selected. Click on a file in the table first to select it, then click Delete.", get_files_list(), "Click on a file row to select it"
    
    try:
        success = vector_store.delete_file(filename)
        if success:
            vector_store.save_index()
            logger.info(f"Successfully deleted file: {filename}")
            return f"✅ Successfully deleted: {filename}", get_files_list(), "File deleted - select another file to view content"
        else:
            logger.error(f"Failed to delete file: {filename}")
            return f"❌ Failed to delete: {filename} (file may not exist)", get_files_list(), ""
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {str(e)}")
        return f"❌ Error deleting file: {str(e)}", get_files_list(), ""