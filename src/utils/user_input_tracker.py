import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

@dataclass
class InputEvent:
    """Base class for all user input events"""
    timestamp: float
    url: str
    event_type: str
    
@dataclass
class MouseClickEvent(InputEvent):
    """Event representing a mouse click"""
    x: int
    y: int
    button: str
    element_selector: Optional[str] = None
    element_text: Optional[str] = None
    
@dataclass
class KeyboardEvent(InputEvent):
    """Event representing keyboard input"""
    key: str
    modifiers: List[str] = field(default_factory=list)
    text: Optional[str] = None
    
@dataclass
class NavigationEvent(InputEvent):
    """Event representing a navigation action"""
    from_url: Optional[str] = None
    to_url: str = ""
    
@dataclass
class ElementSelectionEvent(InputEvent):
    """Event representing an element selection/focus"""
    element_selector: str
    element_text: Optional[str] = None

class UserInputTracker:
    """
    Class to track and record user input events through Chrome DevTools Protocol.
    
    This tracker captures mouse clicks, keyboard inputs, navigation events, and
    element selections during a browser session. It uses CDP to register event
    listeners and stores events in a structured format.
    """
    
    def __init__(self, cdp_client=None):
        """
        Initialize the tracker with a CDP client.
        
        Args:
            cdp_client: Chrome DevTools Protocol client instance
        """
        self.cdp_client = cdp_client
        self.events: List[InputEvent] = []
        self.is_recording = False
        self.current_url = ""
        self._event_listeners = []
        
    async def start_tracking(self, cdp_client=None):
        """
        Start tracking user input events.
        
        Args:
            cdp_client: Optional CDP client to use instead of the one provided at init
        """
        if cdp_client:
            self.cdp_client = cdp_client
            
        if not self.cdp_client:
            logger.error("Cannot start tracking: No CDP client provided")
            return False
            
        if self.is_recording:
            logger.warning("Tracking is already active")
            return True
            
        try:
            # Register CDP event listeners
            await self._register_event_listeners()
            self.is_recording = True
            self.events = []
            logger.info("User input tracking started")
            return True
        except Exception as e:
            logger.error(f"Failed to start tracking: {str(e)}")
            return False
            
    async def stop_tracking(self):
        """Stop tracking user input events and unregister CDP listeners."""
        if not self.is_recording:
            logger.warning("Tracking is not active")
            return
            
        try:
            # Unregister CDP event listeners
            await self._unregister_event_listeners()
            self.is_recording = False
            logger.info("User input tracking stopped")
        except Exception as e:
            logger.error(f"Failed to stop tracking: {str(e)}")
            
    async def _register_event_listeners(self):
        """Register event listeners with the CDP client."""
        if not self.cdp_client:
            return
            
        # Set up event listeners for different input types
        # For mouse events
        self._event_listeners.append(
            await self.cdp_client.session.subscribe(
                "Page.domContentEventFired", self._handle_page_event
            )
        )
        
        # For mouse clicks
        self._event_listeners.append(
            await self.cdp_client.session.subscribe(
                "Input.mousePressed", self._handle_mouse_click
            )
        )
        
        # For keyboard input
        self._event_listeners.append(
            await self.cdp_client.session.subscribe(
                "Input.keyDown", self._handle_key_down
            )
        )
        
        # For navigation
        self._event_listeners.append(
            await self.cdp_client.session.subscribe(
                "Page.frameNavigated", self._handle_navigation
            )
        )
        
    async def _unregister_event_listeners(self):
        """Unregister all CDP event listeners."""
        for listener in self._event_listeners:
            try:
                await listener.dispose()
            except Exception as e:
                logger.error(f"Failed to unregister event listener: {str(e)}")
                
        self._event_listeners = []
        
    async def _handle_page_event(self, event):
        """Handle page content events to update current URL."""
        try:
            # Get current URL
            result = await self.cdp_client.send("Page.getNavigationHistory")
            current_entry = result["entries"][result["currentIndex"]]
            self.current_url = current_entry["url"]
        except Exception as e:
            logger.error(f"Failed to process page event: {str(e)}")
            
    async def _handle_mouse_click(self, event):
        """Handle mouse click events from CDP."""
        if not self.is_recording:
            return
            
        try:
            # Get coordinates from the event
            x = event.get("x", 0)
            y = event.get("y", 0)
            button = event.get("button", "left")
            
            # Try to identify the element at this position
            element_info = await self._get_element_at_position(x, y)
            
            click_event = MouseClickEvent(
                timestamp=time.time(),
                url=self.current_url,
                event_type="mouse_click",
                x=x,
                y=y,
                button=button,
                element_selector=element_info.get("selector"),
                element_text=element_info.get("text")
            )
            
            self.events.append(click_event)
            logger.debug(f"Recorded mouse click: {click_event}")
        except Exception as e:
            logger.error(f"Failed to process mouse click: {str(e)}")
            
    async def _handle_key_down(self, event):
        """Handle keyboard events from CDP."""
        if not self.is_recording:
            return
            
        try:
            key = event.get("key", "")
            modifiers = []
            
            if event.get("alt"):
                modifiers.append("alt")
            if event.get("shift"):
                modifiers.append("shift")
            if event.get("ctrl"):
                modifiers.append("ctrl")
            if event.get("meta"):
                modifiers.append("meta")
                
            # For printable characters
            text = event.get("text", None)
            
            key_event = KeyboardEvent(
                timestamp=time.time(),
                url=self.current_url,
                event_type="keyboard_input",
                key=key,
                modifiers=modifiers,
                text=text
            )
            
            self.events.append(key_event)
            logger.debug(f"Recorded key event: {key_event}")
        except Exception as e:
            logger.error(f"Failed to process keyboard event: {str(e)}")
            
    async def _handle_navigation(self, event):
        """Handle navigation events from CDP."""
        if not self.is_recording:
            return
            
        try:
            frame = event.get("frame", {})
            url = frame.get("url", "")
            
            if not url or url == self.current_url:
                return
                
            nav_event = NavigationEvent(
                timestamp=time.time(),
                url=url,
                event_type="navigation",
                from_url=self.current_url,
                to_url=url
            )
            
            self.current_url = url
            self.events.append(nav_event)
            logger.debug(f"Recorded navigation: {nav_event}")
        except Exception as e:
            logger.error(f"Failed to process navigation event: {str(e)}")
            
    async def _get_element_at_position(self, x, y) -> Dict[str, Any]:
        """
        Get information about the element at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Dictionary with element information (selector, text)
        """
        try:
            if not self.cdp_client:
                return {}
                
            # Use CDP to get the element at the position
            node_id_result = await self.cdp_client.send(
                "DOM.getNodeForLocation", {"x": x, "y": y}
            )
            
            if not node_id_result or "nodeId" not in node_id_result:
                return {}
                
            node_id = node_id_result["nodeId"]
            
            # Get element details
            element_result = await self.cdp_client.send(
                "DOM.describeNode", {"nodeId": node_id}
            )
            
            if not element_result or "node" not in element_result:
                return {}
                
            node = element_result["node"]
            
            # Try to get a CSS selector for this element
            selector_result = await self.cdp_client.send(
                "DOM.querySelector", 
                {"nodeId": node_id, "selector": "*"}
            )
            
            # Get the HTML content of the element
            html_result = await self.cdp_client.send(
                "DOM.getOuterHTML", {"nodeId": node_id}
            )
            
            # Extract text content
            text_content = node.get("nodeValue", "")
            if not text_content and "outerHTML" in html_result:
                # Simple text extraction - in a real implementation, 
                # you might want to use a more robust method
                text_content = html_result["outerHTML"]
                # Strip HTML tags for a simple text representation
                # This is a very basic approach
                text_content = text_content.replace("<", " <").replace(">", "> ")
                
            return {
                "selector": f"#{node.get('attributes', {}).get('id', '')} " 
                            f".{node.get('attributes', {}).get('class', '')}",
                "text": text_content[:100] if text_content else None  # Limit length
            }
        except Exception as e:
            logger.error(f"Failed to get element at position: {str(e)}")
            return {}
            
    def get_events(self) -> List[InputEvent]:
        """
        Get all recorded events.
        
        Returns:
            List of InputEvent objects
        """
        return self.events
        
    def export_events_to_json(self) -> str:
        """
        Export all recorded events to a JSON string.
        
        Returns:
            JSON string representation of all events
        """
        event_dicts = [asdict(event) for event in self.events]
        return json.dumps({
            "version": "1.0",
            "timestamp": time.time(),
            "events": event_dicts
        }, indent=2)
        
    def save_events_to_file(self, filepath: str) -> bool:
        """
        Save all recorded events to a file.
        
        Args:
            filepath: Path to save the events JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            json_data = self.export_events_to_json()
            with open(filepath, 'w') as f:
                f.write(json_data)
            logger.info(f"Saved input events to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save events to file: {str(e)}")
            return False
            
    @classmethod
    def load_events_from_file(cls, filepath: str) -> Tuple[List[InputEvent], bool]:
        """
        Load events from a file.
        
        Args:
            filepath: Path to the events JSON file
            
        Returns:
            Tuple of (events list, success boolean)
        """
        events = []
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if "events" not in data:
                logger.error("Invalid event file format: missing 'events' key")
                return [], False
                
            for event_dict in data["events"]:
                event_type = event_dict.get("event_type")
                
                if event_type == "mouse_click":
                    event = MouseClickEvent(**event_dict)
                elif event_type == "keyboard_input":
                    event = KeyboardEvent(**event_dict)
                elif event_type == "navigation":
                    event = NavigationEvent(**event_dict)
                elif event_type == "element_selection":
                    event = ElementSelectionEvent(**event_dict)
                else:
                    # Default to base class if type not recognized
                    event = InputEvent(**{k: v for k, v in event_dict.items() 
                                         if k in ["timestamp", "url", "event_type"]})
                    
                events.append(event)
                
            logger.info(f"Loaded {len(events)} events from {filepath}")
            return events, True
        except Exception as e:
            logger.error(f"Failed to load events from file: {str(e)}")
            return [], False