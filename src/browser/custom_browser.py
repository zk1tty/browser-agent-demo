import asyncio
import pdb
import os
from pathlib import Path
from typing import Optional

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import (
    BrowserContext as PlaywrightBrowserContext,
)
from playwright.async_api import (
    Playwright,
    async_playwright,
)
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
import logging

from src.browser.custom_context import CustomBrowserContext
from src.browser.custom_context_config import CustomBrowserContextConfig as AppBrowserContextConfig

logger = logging.getLogger(__name__)


class CustomBrowser(Browser):

    async def async_init(self):
        playwright = await async_playwright().start()
        self.playwright = playwright
        
        # Check if we need to use persistent context
        if self.config.chrome_instance_path and "Google Chrome" in self.config.chrome_instance_path:
            user_data_dir = None
            if hasattr(self.config, 'extra_chromium_args') and self.config.extra_chromium_args:
                for arg in self.config.extra_chromium_args:
                    if arg.startswith('--user-data-dir='):
                        user_data_dir = arg.split('=')[1]
                        break
            
            if user_data_dir:
                launch_args = [
                    arg for arg in getattr(self.config, 'extra_chromium_args', [])
                    if not arg.startswith('--user-data-dir=')
                ]
                logger.info(f"Launching persistent Chrome context with UserDataDir: {user_data_dir} and args: {launch_args}")
                self._playwright_browser_context_manager = playwright.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=self.config.headless,
                    args=launch_args,
                    channel="chrome"
                )
                self._playwright_browser = await self._playwright_browser_context_manager

            else:
                logger.info(f"Launching new Chrome browser instance with args: {self.config.extra_chromium_args}")
                self._playwright_browser = await playwright.chromium.launch(
                    headless=self.config.headless,
                    args=self.config.extra_chromium_args,
                    channel="chrome"
                )
        else:
            logger.info(f"Launching new default (Chromium) browser instance with args: {self.config.extra_chromium_args}")
            self._playwright_browser = await playwright.chromium.launch(
                headless=self.config.headless,
                args=self.config.extra_chromium_args
            )
        logger.info(f"Playwright browser initialized: {self._playwright_browser}")

    async def new_context(
            self,
            config: AppBrowserContextConfig = AppBrowserContextConfig()
    ) -> "CustomBrowserContext":
        
        if not hasattr(self, '_playwright_browser') or not self._playwright_browser:
            logger.error("Playwright browser not initialized. Call async_init() first.")
            await self.async_init()
            if not hasattr(self, '_playwright_browser') or not self._playwright_browser:
                 raise RuntimeError("Failed to initialize Playwright browser in new_context.")

        if isinstance(self._playwright_browser, PlaywrightBrowserContext):
            logger.warning("Creating new context from an existing persistent PlaywrightBrowserContext. This might indicate an architectural issue if multiple isolated contexts are expected from a persistent launch.")
            playwright_context_to_wrap = self._playwright_browser 
            logger.info(f"Reusing persistent Playwright context: {playwright_context_to_wrap}")

        elif isinstance(self._playwright_browser, PlaywrightBrowser):
            options = {}
            if config.trace_path:
                pass

            if config.save_recording_path:
                options["record_video_dir"] = config.save_recording_path
                options["record_video_size"] = {"width": config.browser_window_size["width"], "height": config.browser_window_size["height"]}

            if not config.no_viewport and config.browser_window_size:
                 options["viewport"] = {"width": config.browser_window_size["width"], "height": config.browser_window_size["height"]}
            else:
                options["no_viewport"] = True
            
            logger.info(f"Creating new Playwright context with options: {options} from PlaywrightBrowser: {self._playwright_browser}")
            playwright_context_to_wrap = await self._playwright_browser.new_context(**options)
        else:
            logger.error(f"_playwright_browser is of unexpected type: {type(self._playwright_browser)}. Cannot create new context.")
            raise TypeError(f"_playwright_browser is neither PlaywrightBrowser nor PlaywrightBrowserContext.")

        # DEBUGGING: Print ID of CustomBrowserContext class object used here
        from src.browser.custom_context import CustomBrowserContext as CBC_in_CustomBrowser # Alias for clarity
        print(f"DEBUG_INIT: ID of CustomBrowserContext class in custom_browser.py: {id(CBC_in_CustomBrowser)}")

        # Create and return our CustomBrowserContext wrapper
        custom_context = CBC_in_CustomBrowser( # Use aliased import for instantiation
            config=config, # Pass the AppBrowserContextConfig instance
            browser=self,
            playwright_context=playwright_context_to_wrap
        )
        print(f"DEBUG_INIT: Type of CREATED context in custom_browser.py: {type(custom_context)}, ID of its type: {id(type(custom_context))}")
        
        if config.trace_path and playwright_context_to_wrap:
            try:
                await playwright_context_to_wrap.tracing.start(screenshots=True, snapshots=True, sources=True)
                logger.info(f"Context tracing started. Saving to host path: {config.trace_path}")
            except Exception as e:
                logger.error(f"Failed to start tracing: {e}")

        return custom_context

    async def close(self):
        if hasattr(self, '_playwright_browser_context_manager') and self._playwright_browser_context_manager is not None:
             logger.info("Closing persistent Playwright context manager.")
             await self._playwright_browser_context_manager.close()
             self._playwright_browser_context_manager = None
        
        if hasattr(self, '_playwright_browser') and self._playwright_browser is not None:
            if isinstance(self._playwright_browser, PlaywrightBrowserContext):
                 logger.info("Closing PlaywrightBrowserContext (likely persistent context).")
                 await self._playwright_browser.close()
            elif isinstance(self._playwright_browser, PlaywrightBrowser):
                 if self._playwright_browser.is_connected():
                     logger.info("Closing PlaywrightBrowser.")
                     await self._playwright_browser.close()
                 else:
                     logger.info("PlaywrightBrowser is not connected or already closed.")
            else:
                 logger.info(f"_playwright_browser ({type(self._playwright_browser)}) is not a PlaywrightBrowser or PlaywrightBrowserContext that can be closed here, or is already closed.")
            self._playwright_browser = None
        
        if hasattr(self, 'playwright') and self.playwright is not None:
            logger.info("Stopping Playwright.")
            await self.playwright.stop()
            self.playwright = None
        logger.info("CustomBrowser closed.")