"""
File Operations for the UI
Handles document upload, file management, and file browser functionality
"""

import gradio as gr
from services.vector_store import vector_store
from services.logger import logger
import os
def upload_documents(files):
    """Upload and index multiple documents"""
    if not files:
        return "No files uploaded"
    
    results = []
    
    for file in files:
        try:
            # Basic file validation
            filename = os.path.basename(file.name)
            if not filename.endswith(('.pdf', '.txt', '.docx', '.xlsx')):
                results.append(f"✗ Unsupported file type: {filename}")
                continue
                
            # Read file content from the temporary path
            with open(file.path, 'rb') as f:
                content = f.read()
            
            # Add to vector store
            vector_store.add_file(content, filename)
            
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