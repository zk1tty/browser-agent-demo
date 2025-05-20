print("USING USER_INPUT_FUNCTIONS", __file__)
import os
import json
import logging
import asyncio
import glob
import time
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr
from pathlib import Path
from src.browser.custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)

# Global set by webui when a CustomBrowserContext is available
_browser_context = None  # type: Optional[CustomBrowserContext]

# ------------------------------------------------------------------
# API expected by webui.py
# ------------------------------------------------------------------

def set_browser_context(ctx):
    """Called by webui after creating or re‑using a CustomBrowserContext."""
    global _browser_context
    _browser_context = ctx
    logger.debug("Browser context set in user_input_functions: %s", ctx)

def list_input_trace_files(directory_path_str: str) -> List[dict]:
    """Lists input trace files from the specified directory."""
    files_info = []
    try:
        directory = Path(directory_path_str)
        if not directory.is_dir():
            logger.warning(f"Directory not found or not a directory: {directory_path_str}")
            return files_info

        for fp in sorted(directory.glob("*.jsonl")):
            try:
                size_kb = fp.stat().st_size / 1024
                event_count = 0
                with fp.open('r') as f:
                    for _ in f:
                        event_count += 1
                files_info.append({
                    "path": str(fp),
                    "name": fp.name,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fp.stat().st_ctime)),
                    "size": f"{size_kb:.1f} KB",
                    "events": event_count,
                })
            except Exception as e:
                logger.debug("Skipping file %s due to error: %s", fp, e)
    except Exception as e:
        logger.error(f"Error listing trace files in {directory_path_str}: {e}")
    return files_info

async def replay_input_trace(path: str, speed: float = 1.0) -> bool:
    """Entry point used by webui's 'Replay Selected Trace' button."""
    logger.debug("Entered user_input_functions.replay_input_trace – Path: %s", path)
    if _browser_context is None:
        logger.error("No browser context set before replay_input_trace call")
        return False
    try:
        if not isinstance(_browser_context, CustomBrowserContext):
            logger.error(f"Browser context is not a CustomBrowserContext instance: {type(_browser_context)}")
            return False

        ok = await _browser_context.replay_input_events(path, speed=speed, keep_open=True)
        return ok
    except Exception as e:
        logger.exception("Replay failed during replay_input_trace for path %s: %s", path, e)
        return False

async def start_input_tracking() -> tuple[str, bool, str]:
    """
    Start tracking user input events.
    
    Returns:
        Tuple[str, bool, str]: Status message, enable/disable for tracking button, path for file selection
    """
    global _browser_context
    
    if not _browser_context:
        return "No active browser session. Please start a browser session first.", False, ""
        
    try:
        if not isinstance(_browser_context, CustomBrowserContext):
            logger.error(f"Cannot start tracking: _browser_context is not a CustomBrowserContext: {type(_browser_context)}")
            return "Internal error: Browser context not configured correctly.", False, ""

        await _browser_context.start_input_tracking()
        return "User input tracking initiated successfully.", True, ""
    except Exception as e:
        logger.error(f"Error calling _browser_context.start_input_tracking: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", False, ""

async def stop_input_tracking() -> tuple[str, Dict[str, Any], Dict[str, Any], str | None, Dict[str, Any] | None]:
    """
    Stop tracking user input events.
    
    Returns:
        Tuple for Gradio output: Status message, start_btn_update, stop_btn_update, trace_file_path, trace_info_json
    """
    global _browser_context
    
    start_btn_update = gr.update(value="▶️ Start Recording", interactive=True)
    stop_btn_update = gr.update(value="⏹️ Stop Recording", interactive=False)
    default_trace_info = {"message": "No trace information available."}

    if not _browser_context:
        return "No active browser session.", start_btn_update, stop_btn_update, None, default_trace_info
    
    try:
        if not isinstance(_browser_context, CustomBrowserContext):
            logger.error(f"Cannot stop tracking: _browser_context is not a CustomBrowserContext: {type(_browser_context)}")
            return "Internal error: Browser context not configured correctly.", start_btn_update, stop_btn_update, None, default_trace_info

        filepath = await _browser_context.stop_input_tracking()
        if filepath:
            trace_info = get_file_info(filepath)
            return f"Input tracking stopped. Trace saved to: {filepath}", start_btn_update, stop_btn_update, filepath, trace_info
        else:
            return "Input tracking stopped. No trace file was saved.", start_btn_update, stop_btn_update, None, default_trace_info
    except Exception as e:
        logger.error(f"Error calling _browser_context.stop_input_tracking: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", start_btn_update, stop_btn_update, None, default_trace_info

def get_file_info(trace_file_path: str) -> dict[str, any]:
    global _browser_context
    try:
        if not trace_file_path or not isinstance(trace_file_path, str):
            return {"error": "Invalid or no trace file path provided."}

        if not os.path.exists(trace_file_path):
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

        return {
            "file_path": trace_file_path,
            "total_events_counted": event_count,
            "message": "Basic info loaded."
        }
    except Exception as e:
        logger.error(f"Error getting file info for {trace_file_path}: {str(e)}")
        return {"error": str(e), "file_path": trace_file_path}