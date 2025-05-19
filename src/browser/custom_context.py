import json
import logging
import os
import asyncio
from typing import Optional, Dict, Any

from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
from src.utils.user_input_tracker import UserInputTracker, NavigationEvent, MouseClickEvent, KeyboardEvent, InputEvent

logger = logging.getLogger(__name__)


class CustomBrowserContext(BrowserContext):
    def __init__(
            self,
            browser: "Browser",
            config: BrowserContextConfig = BrowserContextConfig(),
            playwright_context=None
    ):
        super().__init__(browser=browser, config=config)
        self.playwright_context = playwright_context
        self.input_tracker: Optional[UserInputTracker] = None
        self.tracking_enabled = False
        self.tracking_save_path = getattr(config, 'save_input_tracking_path', "./tmp/input_tracking")
        
    @property
    def pages(self):
        if self.playwright_context:
            return self.playwright_context.pages
        return []
        
    async def start_user_input_tracking(self) -> bool:
        """
        Start tracking user input events through CDP.
        
        Returns:
            bool: True if tracking started successfully, False otherwise
        """
        if self.tracking_enabled:
            logger.warning("User input tracking is already enabled")
            return True
            
        try:
            # Create directory for saving tracks if it doesn't exist
            if self.tracking_save_path:
                os.makedirs(self.tracking_save_path, exist_ok=True)
                
            # Make sure we have a Playwright context available
            if not self.playwright_context:
                logger.error("Playwright context not available for input tracking")
                return False
                
            # Get first page for CDP access
            if not self.pages:
                logger.error("No pages available in Playwright context for input tracking. Ensure a page is open or will be opened.")
                return False
                
            first_page = self.pages[0]
            # CDP session is now presumably created and managed within UserInputTracker
            # using the context and page provided to its constructor.
            # OLD: cdp_client = await first_page.context.new_cdp_session(first_page)
            
            # Initialize and start the tracker according to the new constructor signature
            self.input_tracker = UserInputTracker(
                context=first_page.context, 
                page=first_page, 
                cdp_client=first_page.context # Passing context as cdp_client as per instruction
            )
            # OLD: result = await self.input_tracker.start_tracking(cdp_client=cdp_client, page=first_page)
            # NEW: start_tracking is called without arguments, UserInputTracker uses its initialized state.
            result = await self.input_tracker.start_tracking()
            
            if result:
                self.tracking_enabled = True
                logger.info("User input tracking started")
                return True
            else:
                logger.error("Failed to start user input tracking")
                return False
                
        except Exception as e:
            logger.error(f"Error starting user input tracking: {str(e)}")
            return False
            
    async def stop_user_input_tracking(self, save_to_file: bool = True) -> Optional[str]:
        """
        Stop tracking user input events and optionally save to file.
        
        Args:
            save_to_file: Whether to save tracking data to file
            
        Returns:
            Optional[str]: Path to the saved tracking file if saved, None otherwise
        """
        if not self.tracking_enabled or not self.input_tracker:
            logger.warning("User input tracking is not enabled or tracker not initialized")
            return None
            
        try:
            await self.input_tracker.stop_tracking()
            self.tracking_enabled = False
            
            if save_to_file and self.tracking_save_path:
                import time # Keep import local if only used here
                timestamp = int(time.time())
                # Ensure tracking_save_path exists (it should have been created in start_tracking)
                os.makedirs(self.tracking_save_path, exist_ok=True)
                filepath = os.path.join(self.tracking_save_path, f"manual_record_{timestamp}.jsonl")
                
                jsonl_data = self.input_tracker.export_events_to_jsonl()
                try:
                    with open(filepath, 'w') as f:
                        f.write(jsonl_data)
                    logger.info(f"Saved user input tracking to {filepath}")
                    return filepath
                except IOError as e:
                    logger.error(f"Failed to write tracking data to file {filepath}: {e}")
                    return None # Failed to save
                    
            logger.info("User input tracking stopped. Data not saved to file as per request or path issue.")
            return None # Tracking stopped, but not saved or no path
            
        except Exception as e:
            logger.error(f"Error stopping user input tracking: {str(e)}")
            return None
            
    async def get_tracking_events_json(self) -> str:
        """
        Get the current tracking events as JSON string.
        
        Returns:
            str: JSON representation of tracked events
        """
        if not self.input_tracker:
            return json.dumps({"error": "No input tracker available"})
            
        return self.input_tracker.export_events_to_json()
            
    async def close(self) -> None:
        """Override close to ensure we stop tracking before closing"""
        if self.tracking_enabled and self.input_tracker:
            await self.stop_user_input_tracking()
            
        await super().close()
        
    async def replay_input_events(self, events_file_path: str, speed: float = 1.0) -> bool:
        """
        Replay recorded input events from a trace file using TraceReplayer.
        
        Args:
            events_file_path: Path to the events JSONL file
            speed: Playback speed multiplier
            
        Returns:
            bool: True if replay successful, False otherwise
        """
        logger.info(f"Attempting to replay trace file: {events_file_path} at speed {speed}x")
        if not self.pages:
            logger.error("No pages available for event replay. Cannot proceed.")
            return False
        
        first_page = self.pages[0] # Use the first available page for replay
        
        try:
            # Ensure these are imported; consider moving to top-level if not causing circular deps
            from src.utils.replayer import TraceReplayer, load_trace, Drift 
            
            trace_events = load_trace(events_file_path)
            if not trace_events:
                logger.error(f"No events found in trace file: {events_file_path}")
                return False

            # Get the URL from the first event to set the initial state
            first_recorded_event = trace_events[0]
            initial_url = first_recorded_event.get("url")

            if initial_url and first_page.url.rstrip('/') != initial_url.rstrip('/'):
                logger.info(f"Initial URL of trace ({initial_url}) differs from current page URL ({first_page.url}). Navigating...")
                try:
                    await first_page.goto(initial_url, wait_until="networkidle", timeout=15000) # Added timeout
                    logger.info(f"Successfully navigated to initial URL: {initial_url}")
                except Exception as nav_exc:
                    logger.error(f"Failed to navigate to initial URL {initial_url} for replay: {nav_exc}")
                    return False # Cannot proceed if initial navigation fails
            
            # Correctly instantiate TraceReplayer with page and trace_events
            replayer = TraceReplayer(page=first_page, trace=trace_events)
            # Call play with the speed parameter
            await replayer.play(speed=speed) 
            logger.info(f"Successfully replayed trace file: {events_file_path}")
            return True
        except FileNotFoundError:
            logger.error(f"Trace file not found for replay: {events_file_path}")
            return False
        except Drift as d: # Catch Drift specifically to log its details
            logger.error(f"Drift detected during replay of {events_file_path}: {str(d)}") # Ensure this line uses str(d)
            if hasattr(d, 'event') and d.event: # Check if event attribute exists
                 logger.error(f"   Drift occurred at event: {json.dumps(d.event)}")
            return False
        except Exception as e:
            import traceback # Keep local for this specific exception handler
            logger.error(f"Error during event replay of {events_file_path}: {str(e)}\\n{traceback.format_exc()}")
            return False

# Example of how config might be used for this context if passed directly
# config = BrowserContextConfig(save_input_tracking_path="./my_traces")
# context = CustomBrowserContext(browser_instance, config=config)