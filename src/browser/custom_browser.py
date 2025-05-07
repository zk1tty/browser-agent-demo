import asyncio
import pdb
import os

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import (
    BrowserContext as PlaywrightBrowserContext,
)
from playwright.async_api import (
    Playwright,
    async_playwright,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
import logging

from .custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)


class CustomBrowser(Browser):

    async def new_context(
            self,
            config: BrowserContextConfig = BrowserContextConfig()
    ) -> CustomBrowserContext:
        # Log configuration status
        logger.info("==========Browser Configuration Status:==========")
        logger.info(f"Chrome Instance Path: {self.config.chrome_instance_path}")
        logger.info(f"CDP URL: {self.config.cdp_url}")
        logger.info(f"Extra Chromium Args: {getattr(self.config, 'extra_chromium_args', [])}")
        logger.info(f"Config Object: {self.config}")
        
        # Add Chrome-specific configuration: https://playwright.dev/python/docs/chrome-extensions
        if self.config.chrome_instance_path:
            # Ensure we're using Chrome and not Chromium
            if "Google Chrome" in self.config.chrome_instance_path:
                logger.info("Using Google Chrome browser")
                # Add Chrome-specific flags
                if not hasattr(self.config, 'extra_chromium_args'):
                    self.config.extra_chromium_args = []
                chrome_flags = [
                    "--disable-features=NewTabPage",  # Prevent loading New Tab Page in non-browser-tab context
                    "--profile-directory=Default",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--user-data-dir=${HOME}/Library/Application Support/Google/Chrome",  # Use default Chrome profile
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-web-security",
                    "--disable-site-isolation-trials",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process,LavaMoat",
                    "--disable-extensions",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-software-rasterizer",
                    "--disable-features=BlockInsecurePrivateNetworkRequests",
                    "--disable-features=CrossSiteDocumentBlockingIfIsolating",
                    "--disable-features=CrossSiteDocumentBlockingAlways"
                ]
                self.config.extra_chromium_args.extend(chrome_flags)
                logger.info(f"Added Chrome-specific flags: {chrome_flags}")
            else:
                logger.warning("Not using Google Chrome browser - configuration may not work as expected")
        
        return CustomBrowserContext(config=config, browser=self)
