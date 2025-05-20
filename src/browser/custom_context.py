from __future__ import annotations
import logging, time, asyncio
from pathlib import Path
from typing import Optional

from browser_use.browser.browser import Browser # Import Browser for type hinting
from browser_use.browser.context import BrowserContext, BrowserContextConfig # Import base class and its config
from src.browser.custom_context_config import CustomBrowserContextConfig as AppCustomBrowserContextConfig # Specific config for this app

from src.utils.user_input_tracker import UserInputTracker
from src.utils.replayer import TraceReplayer, Drift, load_trace

logger = logging.getLogger(__name__)

class CustomBrowserContext(BrowserContext): # Inherit from BrowserContext
    """Wrapper around a Playwright BrowserContext to add record/replay helpers."""

    # ---------------- construction helpers -----------------

    def __init__(self, pw_context, browser: 'Browser', config: AppCustomBrowserContextConfig = AppCustomBrowserContextConfig()): # Add browser and config
        super().__init__(browser=browser, config=config) # Call super with browser and config
        self._ctx = pw_context                          # Playwright BrowserContext
        # self._pages = pw_context.pages # pages is a dynamic property
        self.input_tracker: Optional[UserInputTracker] = None
        # self.save_dir is now handled by base class if config is used correctly, or can be specific here
        # For now, let specific save_dir override if base doesn't use it from config the same way.
        self.save_dir = Path(getattr(config, 'save_input_tracking_path', "./tmp/input_tracking")) 
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._dom_bridge_initialized_on_context = False # New instance flag

        # Removed: asyncio.create_task(self._ensure_dom_bridge())
        
    @property
    def playwright_context(self):
        return self._ctx

    @classmethod
    def from_existing(cls, pw_context, browser: 'Browser', config: AppCustomBrowserContextConfig = AppCustomBrowserContextConfig()): # Add browser and config
        # This method creates an instance, so it needs to provide what __init__ expects.
        # The base BrowserContext does not have from_existing, so this is specific.
        # It should call cls(pw_context, browser, config)
        return cls(pw_context=pw_context, browser=browser, config=config)

    # ---------------- private bootstrap -------------------

    BINDING = "__uit_relay"

    async def _ensure_dom_bridge(self):
        # Check instance flag first, then context attribute as a fallback/secondary check
        if self._dom_bridge_initialized_on_context or getattr(self._ctx, "_uit_ready", False):
            logger.debug("DOM bridge already considered installed on context %s. Skipping.", id(self._ctx))
            return
        
        from src.utils.user_input_tracker import UserInputTracker as UITracker # Alias for clarity
        js_template_for_context = UITracker._JS_TEMPLATE 
        
        try:
            logger.debug(f"Attempting to expose binding '{self.BINDING}' on context {id(self._ctx)}.")
            await self._ctx.expose_binding(self.BINDING, self._on_binding_wrapper)
            logger.debug(f"Successfully exposed binding '{self.BINDING}' on context {id(self._ctx)}.")
            
            # Yield to event loop to ensure binding registration completes before init script runs
            await asyncio.sleep(0)
            
            logger.debug(f"Attempting to add init script to context {id(self._ctx)}.")
            # Log the script content before injection
            script_to_inject = js_template_for_context.format(binding=self.BINDING)
            logger.debug(f"CUSTOM_CONTEXT: About to inject script (first 120 chars): {script_to_inject[:120]}")
            await self._ctx.add_init_script(script_to_inject)
            logger.debug(f"Successfully added init script to context {id(self._ctx)}.")

            # Quick patch to confirm injection via evaluate
            if self._ctx.pages: # Check if there are any pages in the context
                try:
                    await self._ctx.pages[0].evaluate("console.log('[CUSTOM_CONTEXT] Test eval after add_init_script OK')")
                    logger.debug("[CUSTOM_CONTEXT] Test eval after add_init_script executed on page 0.")
                except Exception as eval_err:
                    logger.error(f"[CUSTOM_CONTEXT] Error during test eval on page 0: {eval_err}")
            else:
                logger.warning("[CUSTOM_CONTEXT] No pages in context to run test eval after add_init_script.")
            
            setattr(self._ctx, "_uit_ready", True) # Mark on Playwright context
            self._dom_bridge_initialized_on_context = True # Mark on CustomBrowserContext instance
            logger.debug("DOM bridge successfully installed on context %s", id(self._ctx))
            
        except Exception as e: # Catch Playwright's Error for already registered or other issues
            if "already registered" in str(e).lower():
                logger.warning(f"Binding '{self.BINDING}' or script already registered on context {id(self._ctx)}, but flags were not set. Marking as ready now. Error: {e}")
                setattr(self._ctx, "_uit_ready", True) 
                self._dom_bridge_initialized_on_context = True
            else:
                logger.error(f"Failed to install DOM bridge on context {id(self._ctx)}: {e}", exc_info=True)
                # Do not set flags to true if it was a different error, so it might be retried if appropriate.
                # Or, re-raise if this is considered a fatal error for context setup.
                raise # Re-raise other errors for now

    # ---------------- binding passthrough ------------------
    
    async def _on_binding_wrapper(self, source, payload):
        page = source.get("page") 
        if not page:
            logger.error("Page not found in binding source. Cannot initialize or use tracker.")
            return

        if not self.input_tracker:
            logger.debug(f"Lazy-initializing UserInputTracker for page: {page.url} (context: {id(self._ctx)})")
            self.input_tracker = UserInputTracker(context=self._ctx, page=page) 
            self.input_tracker.is_recording = True 
            self.input_tracker.current_url = page.url 
            
            if self.input_tracker and self.input_tracker.context and hasattr(self.input_tracker, '_setup_page_listeners'): # Extra guard for linter
                logger.debug(f"CONTEXT_EVENT: Attaching context-level 'page' event listener in CustomBrowserContext for context {id(self._ctx)}")
                self.input_tracker.context.on("page", 
                    lambda p: asyncio.create_task(self._log_and_setup_page_listeners(p)))
                
                await self.input_tracker._setup_page_listeners(page) 
            elif not (self.input_tracker and self.input_tracker.context):
                logger.error("Input tracker or its context not set after initialization during listener setup.")
            elif not hasattr(self.input_tracker, '_setup_page_listeners'):
                 logger.error("_setup_page_listeners method not found on input_tracker instance.")

        if self.input_tracker:
            await self.input_tracker._on_dom_event(source, payload) 
        else:
            # This case should ideally not be reached if logic above is correct
            logger.error("Input tracker somehow still not initialized in _on_binding_wrapper before passing event.")

    # New helper method to log before calling _setup_page_listeners
    async def _log_and_setup_page_listeners(self, page_object):
        logger.debug(f"CONTEXT_EVENT: Context 'page' event fired! Page URL: {page_object.url}, Page Object ID: {id(page_object)}. Calling _setup_page_listeners.")
        if self.input_tracker: # Ensure input_tracker still exists
            await self.input_tracker._setup_page_listeners(page_object)
        else:
            logger.error("CONTEXT_EVENT: self.input_tracker is None when _log_and_setup_page_listeners was called.")

    # ---------------- recording API -----------------------

    async def start_input_tracking(self): 
        await self._ensure_dom_bridge()

        current_pages = self.pages
        page_to_use = None

        if current_pages:
            # Prefer non-devtools, non-chrome internal pages
            content_pages = [
                p for p in current_pages 
                if p.url and 
                   not p.url.startswith("devtools://") and 
                   not p.url.startswith("chrome://") and 
                   not p.url.startswith("about:") # Also exclude about:blank from being primary unless it's the only one after filtering
            ]
            if content_pages:
                page_to_use = content_pages[0]
                logger.debug(f"Using existing content page for tracking: {page_to_use.url}")
            else:
                # If no "ideal" content pages, check if there are any pages at all (e.g. only about:blank or chrome://newtab)
                non_devtools_pages = [p for p in current_pages if p.url and not p.url.startswith("devtools://")]
                if non_devtools_pages:
                    page_to_use = non_devtools_pages[0]
                    logger.debug(f"No ideal content pages. Using first non-devtools page: {page_to_use.url}")
                else:
                    logger.warning("No suitable (non-devtools) pages found. Creating a new page.")
                    page_to_use = await self.new_page()
                    if page_to_use: await page_to_use.goto("about:blank") # Navigate to a blank page
        else:
            logger.debug("No pages in current context. Creating a new page.")
            page_to_use = await self.new_page()
            if page_to_use: await page_to_use.goto("about:blank") # Navigate to a blank page
        
        if not page_to_use:
            logger.error("Could not get or create a suitable page for input tracking. Tracking will not start.")
            return

        if not self.input_tracker: # Initialize UserInputTracker if it doesn't exist
            logger.debug(f"Initializing UserInputTracker for page: {page_to_use.url}")
            self.input_tracker = UserInputTracker(context=self._ctx, page=page_to_use)
            # The UserInputTracker.start_tracking() will call _setup_page_listeners for this page_to_use
            await self.input_tracker.start_tracking() 
        elif not self.input_tracker.is_recording: # If tracker exists but not recording
            logger.debug(f"Re-activating recording on existing input tracker. Ensuring it targets page: {page_to_use.url}")
            self.input_tracker.page = page_to_use # Explicitly update the page on the existing tracker
            self.input_tracker.current_url = page_to_use.url
            await self.input_tracker.start_tracking() # This will call _setup_page_listeners for the (potentially new) page
        else: # Tracker exists and is recording
            if self.input_tracker.page != page_to_use:
                if page_to_use: # Explicitly check page_to_use is not None here
                    logger.warning(f"Input tracker is active but on page {self.input_tracker.page.url if self.input_tracker.page else 'None'}. Forcing switch to {page_to_use.url}")
                    self.input_tracker.page = page_to_use
                    self.input_tracker.current_url = page_to_use.url
                    await self.input_tracker.start_tracking() # Re-run to ensure listeners are on this page
                else:
                    # This case should ideally not be reached due to earlier checks, but as a safeguard:
                    logger.error("Input tracker is active, but the determined page_to_use is None. Cannot switch tracker page.")
            else: # self.input_tracker.page == page_to_use
                if page_to_use: # page_to_use should not be None here if it matches a valid tracker page
                    logger.debug(f"Input tracking is already active and on the correct page: {page_to_use.url}")
                else: # Should be an impossible state if self.input_tracker.page was not None
                    logger.error("Input tracking is active, but page_to_use is None and matched self.input_tracker.page. Inconsistent state.")
        
        if page_to_use: # Final log should also be conditional
            logger.debug(f"User input tracking active. Target page: {page_to_use.url}")
        # If page_to_use is None here, an error was logged and function returned earlier.

    async def stop_input_tracking(self):
        if self.input_tracker and self.input_tracker.is_recording:
            await self.input_tracker.stop_tracking() 
            filename = f"manual_record_{int(time.time())}.jsonl"
            path = self.save_dir / filename
            jsonl_data = self.input_tracker.export_events_to_jsonl()
            if jsonl_data.strip():
                path.write_text(jsonl_data)
                logger.info("Saved user input tracking to %s", path)
                return str(path)
            else:
                logger.info("No events recorded, skipping file save.")
                return None
        else:
            logger.warning("Input tracking not active or tracker not initialized, nothing to stop/save.")
            return None


    # ---------------- replay API --------------------------

    async def replay_input_events(self, trace_path: str, speed: float = 2.0, keep_open: bool = True):
        current_pages = self.pages
        page_for_replay = current_pages[0] if current_pages else await self.new_page()
        if not page_for_replay:
            logger.error("Cannot replay events, no page available.")
            return False
            
        trace_data = load_trace(trace_path)
        if not trace_data:
            logger.error(f"Trace file {trace_path} is empty or could not be loaded.")
            return False
        
        rep = TraceReplayer(page_for_replay, trace_data)
        try:
            await rep.play(speed=speed)
            logger.info("Successfully replayed trace file: %s", trace_path)
            return True
        except Drift as d:
            logger.error("Drift detected during replay of %s: %s", trace_path, d)
            return False
        except Exception as e:
            import traceback
            logger.error(f"Unexpected error during replay of {trace_path}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if not keep_open:
                logger.info("Replay finished and keep_open is False. Closing context.")
                await self.close() # Call own close method

    async def close(self):
        logger.info(f"Closing CustomBrowserContext (Playwright context id: {id(self._ctx)}).")
        # Check input_tracker before accessing is_recording
        if hasattr(self, 'input_tracker') and self.input_tracker and self.input_tracker.is_recording:
            logger.info("Input tracking is active, stopping it before closing context.")
            await self.stop_input_tracking()
        
        if self._ctx and not self._ctx.is_closed():
             await self._ctx.close()
        logger.info("CustomBrowserContext closed.")

    @property
    def pages(self):
        if self._ctx:
            try:
                return self._ctx.pages
            except Exception: # Broad exception for now, ideally Playwright-specific error
                # This can happen if the context or browser is closed.
                return []
        return []
    
    async def new_page(self, **kwargs):
        if self._ctx:
            try:
                # Attempting to access pages is a way to check if context is usable
                _ = self._ctx.pages 
                return await self._ctx.new_page(**kwargs)
            except Exception as e: # Catch error if context is closed
                logger.error(f"Playwright context not available or closed when trying to create new page: {e}")
                return None
        logger.error("Playwright context (_ctx) is None, cannot create new page.")
        return None