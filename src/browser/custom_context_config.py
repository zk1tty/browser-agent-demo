import os
from dataclasses import dataclass
from browser_use.browser.context import BrowserContextConfig

@dataclass
class CustomBrowserContextConfig(BrowserContextConfig):
    """Extended BrowserContextConfig with user input tracking settings."""
    # Base fields from parent class are inherited
    
    # User input tracking settings
    enable_input_tracking: bool = False
    save_input_tracking_path: str = "./tmp/input_tracking"
    
    def __post_init__(self):
        """Ensure directory paths are absolute and properly formatted."""
        if self.save_input_tracking_path and not self.save_input_tracking_path.startswith("/"):
            # Convert to absolute path using the current directory if relative
            self.save_input_tracking_path = os.path.abspath(self.save_input_tracking_path)
        
        # Only call parent if it exists
        try:
            super().__post_init__()
        except AttributeError:
            pass