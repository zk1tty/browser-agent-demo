import os
import json
import logging
import asyncio
import glob
import time
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr

logger = logging.getLogger(__name__)

# Avoid circular import by not importing _global_browser_context
# We'll use a global variable that will be set from webui.py instead
_browser_context = None

def set_browser_context(context):
    """Set the browser context to use for functions in this module."""
    global _browser_context
    _browser_context = context

async def start_input_tracking() -> Tuple[str, bool, str]:
    """
    Start tracking user input events.
    
    Returns:
        Tuple[str, bool, str]: Status message, enable/disable for tracking button, path for file selection
    """
    global _browser_context
    
    if not _browser_context:
        return "No active browser session. Please start a browser session first.", False, ""
        
    try:
        result = await _browser_context.start_user_input_tracking()
        if result:
            return "User input tracking started successfully. Interact with the browser to record actions.", True, ""
        else:
            return "Failed to start user input tracking. See logs for details.", False, ""
    except Exception as e:
        logger.error(f"Error starting user input tracking: {str(e)}")
        return f"Error: {str(e)}", False, ""
        
async def stop_input_tracking() -> Tuple[str, bool, Optional[str]]:
    """
    Stop tracking user input events.
    
    Returns:
        Tuple[str, bool, Optional[str]]: Status message, enable/disable for tracking, trace file path
    """
    global _browser_context
    
    if not _browser_context:
        return "No active browser session.", False, None
        
    try:
        filepath = await _browser_context.stop_user_input_tracking(save_to_file=True)
        if filepath:
            return f"Input tracking stopped. Trace saved to: {filepath}", False, filepath
        else:
            return "Input tracking stopped. No trace file was saved.", False, None
    except Exception as e:
        logger.error(f"Error stopping user input tracking: {str(e)}")
        return f"Error: {str(e)}", False, None
        
def list_input_trace_files(directory_path: str = "./tmp/input_tracking") -> List[Dict[str, str]]:
    """
    List all input trace files in the specified directory.
    
    Args:
        directory_path: Directory to search for trace files
        
    Returns:
        List of trace files with metadata
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return []
        
    # Get all JSON files
    trace_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    # Sort files by creation time (newest first)
    trace_files.sort(key=os.path.getctime, reverse=True)
    
    # Create file info list
    file_info = []
    for trace_file in trace_files:
        filename = os.path.basename(trace_file)
        created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(trace_file)))
        file_size = os.path.getsize(trace_file) / 1024  # size in KB
        
        try:
            # Read file to get event count
            with open(trace_file, 'r') as f:
                data = json.load(f)
                event_count = len(data.get("events", []))
        except Exception:
            event_count = "N/A"
            
        file_info.append({
            "path": trace_file,
            "name": filename,
            "created": created_time,
            "size": f"{file_size:.1f} KB",
            "events": event_count
        })
        
    return file_info
    
def get_file_info(trace_file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a trace file.
    
    Args:
        trace_file_path: Path to the trace file
        
    Returns:
        Dictionary with file information
    """
    try:
        if not os.path.exists(trace_file_path):
            return {"error": f"File not found: {trace_file_path}"}
            
        with open(trace_file_path, 'r') as f:
            data = json.load(f)
            
        event_counts = {}
        urls = set()
        
        events = data.get("events", [])
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            url = event.get("url", "")
            if url:
                urls.add(url)
                
        return {
            "total_events": len(events),
            "event_types": event_counts,
            "unique_urls": list(urls),
            "timestamp": data.get("timestamp", "Unknown"),
            "version": data.get("version", "Unknown")
        }
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {"error": str(e)}
        
async def replay_input_trace(trace_file_path: str) -> str:
    """
    Replay a recorded input trace.
    
    Args:
        trace_file_path: Path to the trace file
        
    Returns:
        Status message
    """
    global _browser_context
    
    if not _browser_context:
        return "No active browser session. Please start a browser session first."
        
    if not os.path.exists(trace_file_path):
        return f"File not found: {trace_file_path}"
        
    try:
        result = await _browser_context.replay_input_events(trace_file_path)
        if result:
            return "Input trace replay completed successfully."
        else:
            return "Failed to replay input trace. See logs for details."
    except Exception as e:
        logger.error(f"Error replaying input trace: {str(e)}")
        return f"Error: {str(e)}"

    user_input_functions.set_browser_context(None)