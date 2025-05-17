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
        
async def stop_input_tracking() -> Tuple[str, Any, Any, Optional[str]]:
    """
    Stop tracking user input events.
    
    Returns:
        Tuple for Gradio output: Status message, start_btn_update, stop_btn_update, trace_file_path
    """
    print("--- DEBUG: Entered user_input_functions.stop_input_tracking ---")
    global _browser_context
    
    start_btn_update = gr.update(value="Start Recording", interactive=True)
    stop_btn_update = gr.update(value="Stop Recording", interactive=False) # Default to non-interactive after stop

    if not _browser_context:
        print("--- DEBUG: _browser_context is None. Returning. ---")
        return "No active browser session.", start_btn_update, stop_btn_update, None
    
    print(f"--- DEBUG: _browser_context is type: {type(_browser_context)} ---")
    try:
        print("--- DEBUG: Calling _browser_context.stop_user_input_tracking ---")
        filepath = await _browser_context.stop_user_input_tracking(save_to_file=True)
        print(f"--- DEBUG: filepath from stop_user_input_tracking: {filepath} ---")
        if filepath:
            print(f"--- DEBUG: Filepath exists: {filepath}. Returning success message. ---")
            return f"Input tracking stopped. Trace saved to: {filepath}", start_btn_update, stop_btn_update, filepath
        else:
            print("--- DEBUG: Filepath is None. Returning 'no trace file saved' message. ---")
            return "Input tracking stopped. No trace file was saved.", start_btn_update, stop_btn_update, None
    except Exception as e:
        print(f"--- DEBUG: Exception in stop_input_tracking: {str(e)} ---")
        logger.error(f"Error stopping user input tracking: {str(e)}")
        return f"Error: {str(e)}", start_btn_update, stop_btn_update, None
        
def list_input_trace_files(directory_path: str = "./tmp/input_tracking") -> List[Dict[str, str]]:
    print(f"--- DEBUG list_input_trace_files: Listing from directory: {directory_path} ---")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print("--- DEBUG list_input_trace_files: Directory did not exist, created. Returning empty list. ---")
        return []
        
    trace_files = glob.glob(os.path.join(directory_path, "*.jsonl"))
    print(f"--- DEBUG list_input_trace_files: Found files: {trace_files} ---")
    
    file_info = []
    for trace_file_path_iter in trace_files:
        filename = os.path.basename(trace_file_path_iter)
        created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(trace_file_path_iter)))
        file_size = os.path.getsize(trace_file_path_iter) / 1024  # size in KB
        
        event_count = 0
        try:
            with open(trace_file_path_iter, 'r') as f:
                for line in f:
                    if line.strip(): # Ensure line is not empty
                        event_count += 1
        except Exception:
            event_count = "N/A" # Or handle error more specifically
            
        file_info.append({
            "path": trace_file_path_iter,
            "Name": filename,            
            "Created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(trace_file_path_iter))),
            "Size": f"{os.path.getsize(trace_file_path_iter) / 1024:.1f} KB",
            "Events": event_count # event_count is calculated in list_input_trace_files
        })
    print(f"--- DEBUG list_input_trace_files: Returning file_info: {file_info} ---")
    return file_info
    
def get_file_info(trace_file_path: str) -> Dict[str, Any]:
    print(f"--- DEBUG get_file_info: Getting info for: {trace_file_path}, type: {type(trace_file_path)} ---")
    try:
        if not trace_file_path or not isinstance(trace_file_path, str):
            print("--- DEBUG get_file_info: trace_file_path is None, empty or not a string. ---")
            return {"error": "Invalid or no trace file path provided."}

        if not os.path.exists(trace_file_path):
            print(f"--- DEBUG get_file_info: File not found at path: {trace_file_path} ---")
            return {"error": f"File not found: {trace_file_path}"}
            
        event_count = 0
        first_event_info = "No events found or could not read first event."
        urls = set()
        event_types_summary = {}

        with open(trace_file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    event_count += 1
                    try:
                        event = json.loads(line)
                        if i == 0: # Information from the first event
                            ts = event.get("t", "N/A")
                            ev_type = event.get("type", "N/A")
                            first_event_info = f"First event (dt={ts}ms, type={ev_type})"
                        
                        # Summarize event types
                        current_type = event.get("type", "unknown")
                        event_types_summary[current_type] = event_types_summary.get(current_type, 0) + 1
                        
                        # Collect URLs from navigation events
                        if event.get("type") == "navigation" and "to" in event:
                            urls.add(event["to"])
                        elif "url" in event: # For other event types that might have a URL
                            urls.add(event["url"])
                            
                    except json.JSONDecodeError:
                        if i == 0:
                            first_event_info = "First line is not valid JSON."
                        # Optionally log this error for other lines
                        pass # continue to count lines even if some are not valid JSON

        print(f"--- DEBUG get_file_info: Counted {event_count} events in {trace_file_path} ---")
        # For now, return a simplified structure to see if it fixes the display issue
        # The full parsing can be re-added once the Gradio cache issue is sorted
        return {
            "file_path": trace_file_path,
            "total_events_counted": event_count,
            "message": "Basic info loaded. Full parsing pending cache issue resolution."
        }
    except Exception as e:
        print(f"--- DEBUG get_file_info: Exception: {str(e)} for file {trace_file_path} ---")
        logger.error(f"Error getting file info for {trace_file_path}: {str(e)}")
        return {"error": str(e), "file_path": trace_file_path}
        
async def replay_input_trace(trace_file_path: str) -> str:
    """
    Replay a recorded input trace.
    
    Args:
        trace_file_path: Path to the trace file
        
    Returns:
        Status message
    """
    print(f"--- DEBUG: Entered user_input_functions.replay_input_trace --- Path: {trace_file_path}, Type: {type(trace_file_path)} ---")
    global _browser_context
    
    if not _browser_context:
        print("--- DEBUG: replay_input_trace - _browser_context is None. Returning. ---")
        return "No active browser session. Please start a browser session first."
        
    if not trace_file_path or not isinstance(trace_file_path, str):
        print(f"--- DEBUG: replay_input_trace - Invalid trace_file_path: {trace_file_path}. Returning. ---")
        return "No trace file path provided or path is not a string."

    if not os.path.exists(trace_file_path):
        print(f"--- DEBUG: replay_input_trace - File not found at path: {trace_file_path}. Returning. ---")
        return f"File not found: {trace_file_path}"
        
    try:
        print(f"--- DEBUG: replay_input_trace - Calling _browser_context.replay_input_events for path: {trace_file_path} ---")
        result = await _browser_context.replay_input_events(trace_file_path)
        if result:
            print("--- DEBUG: replay_input_trace - Replay successful. ---")
            return "Input trace replay completed successfully."
        else:
            print("--- DEBUG: replay_input_trace - Replay failed. ---")
            return "Failed to replay input trace. See logs for details."
    except Exception as e:
        print(f"--- DEBUG: replay_input_trace - Exception: {str(e)} ---")
        logger.error(f"Error replaying input trace: {str(e)}")
        return f"Error: {str(e)}"