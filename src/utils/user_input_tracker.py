print("USING TRACKER", __file__)
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

# =====================
# Dataclass definitions
# =====================

@dataclass
class InputEvent:
    timestamp: float
    url: str
    event_type: str

@dataclass
class MouseClickEvent(InputEvent):
    x: int
    y: int
    button: str
    selector: str
    text: str | None = None
    modifiers: List[str] = field(default_factory=list)

@dataclass
class KeyboardEvent(InputEvent):
    key: str
    code: str
    selector: str
    modifiers: List[str] = field(default_factory=list)

@dataclass
class NavigationEvent(InputEvent):
    from_url: Optional[str] = None
    to_url: str = ""

# =========================================================
# Main tracker class
# =========================================================

class UserInputTracker:
    """Tracks mouse, keyboard, and navigation events via Playwright + CDP."""

    BINDING = "__uit_relay"

    _JS_TEMPLATE = """
(function () {{
  console.log('[UIT SCRIPT] Attempting to run on URL:', location.href, 'Is top window:', window.top === window, 'Timestamp:', Date.now());

  if (window.top !== window) {{
    console.log('[UIT SCRIPT] EXIT (not top window) on URL:', location.href);
    return;
  }}

  // Global guard on window.top itself, persists across navigations in the same tab.
  if (window.top.__uit_global_listeners_attached) {{
    console.log('[UIT SCRIPT] GUARDED (globally, listeners already attached by a previous script instance in this tab) on URL:', location.href);
    return; 
  }}

  // If we reach here, we are in the top window, and this is the first script instance in this tab to pass the global guard.
  console.log('[UIT SCRIPT] PASSED GLOBAL GUARD: Marking tab as having listeners and proceeding to setup for URL:', location.href);
  window.top.__uit_global_listeners_attached = true; // Set the persistent global flag

  const binding = '{binding}'; 

  function actualListenerSetup() {{
    console.log('[UIT SCRIPT] actualListenerSetup: Called for document of URL:', document.location.href);

    function smartSelector(el) {{
        // Ensure documentElement is available before trying to use it or querySelector
        if (!document || !document.documentElement) {{
            console.warn('[UIT SCRIPT] smartSelector: documentElement not available for URL:', document.location.href);
            return '';
        }}
        if (!el || el.nodeType !== 1) return '';
        if (el.id) return '#' + CSS.escape(el.id);
        const attrs = ['data-testid','aria-label','role','name','placeholder'];
        for (const a of attrs) {{
            const v = el.getAttribute(a);
            if (v) {{
                const sel_val = el.localName + '[' + a + '=\\"' + CSS.escape(v) + '\\"]';
                try {{ if (document.querySelectorAll(sel_val).length === 1) return sel_val; }} catch (e) {{/*ignore*/}}
            }}
        }}
        let path = '', depth = 0, node = el;
        while (node && node.nodeType === 1 && node !== document.documentElement && depth < 10) {{
            let seg = node.localName;
            if (node.parentElement) {{
                const children = node.parentElement.children;
                const sib = Array.from(children || []).filter(s => s.localName === seg);
                if (sib.length > 1) {{
                    const idx = sib.indexOf(node);
                    if (idx !== -1) {{ seg += ':nth-of-type(' + (idx + 1) + ')'; }}
                }}
            }}
            path = path ? seg + '>' + path : seg;
            try {{ if (document.querySelectorAll(path).length === 1) return path; }} catch (e) {{/*ignore*/}}
            if (!node.parentElement) break; 
            node = node.parentElement; depth++;
        }}
        return path || (el.localName ? el.localName : ''); // Fallback to localName if path is empty
    }}

    const send = (type, e) => {{
      if (e.repeat) return;

      const makePayload = () => ({{
        type, ts: Date.now(), url: document.location.href, // Use document.location.href for current frame
        selector: smartSelector(e.target || document.activeElement),
        x: e.clientX ?? null, y: e.clientY ?? null, button: e.button ?? null,
        key: e.key ?? null, code: e.code ?? null,
        modifiers: {{alt:e.altKey,ctrl:e.ctrlKey,shift:e.shiftKey,meta:e.metaKey}},
        text: (type === 'mousedown' && e.target?.innerText) ? (e.target.innerText || '').trim().slice(0,50) : ((e.target?.value || '').trim().slice(0,50) || null)
      }});
                                  
      if (typeof window[binding] === 'function') {{
        window[binding](makePayload());
      }} else {{
        console.warn('[UIT SCRIPT] send: Binding not ready for', type, 'on URL:', document.location.href, 'Retrying in 50ms.');
        setTimeout(() => {{
            if (typeof window[binding] === 'function') {{
                console.log('[UIT SCRIPT] send: Binding ready after delay for', type, 'on URL:', document.location.href);
                window[binding](makePayload());
            }} else {{
                console.error('[UIT SCRIPT] send: Binding STILL not ready after delay for', type, 'on URL:', document.location.href);
            }}
        }}, 50);
      }}
    }};

    // These listeners are attached to the current document (which is window.top.document here)
    document.addEventListener('mousedown', e => send('mousedown', e), true);
    document.addEventListener('keydown',   e => send('keydown',   e), true);
    console.log('[UIT SCRIPT] actualListenerSetup: Event listeners ATTACHED to document of URL:', document.location.href);
  }}

  function deferredSetupCaller() {{
    console.log('[UIT SCRIPT] deferredSetupCaller: Checking binding and document state for URL:', document.location.href);
    if (typeof window[binding] === 'function') {{
      console.log('[UIT SCRIPT] Binding found immediately for document:', document.location.href);
      if (document.readyState === 'loading') {{
        console.log('[UIT SCRIPT] Document still loading, deferring actualListenerSetup for URL:', document.location.href);
        document.addEventListener('DOMContentLoaded', actualListenerSetup);
      }} else {{
        console.log('[UIT SCRIPT] Document already loaded (or interactive), calling actualListenerSetup for URL:', document.location.href);
        actualListenerSetup();
      }}
    }} else {{
      console.log('[UIT SCRIPT] Binding not immediately found for document:', document.location.href, '. Will check again in 10ms.');
      setTimeout(() => {{
        if (typeof window[binding] === 'function') {{
          console.log('[UIT SCRIPT] Binding found after 10ms delay for document:', document.location.href);
          if (document.readyState === 'loading') {{
            console.log('[UIT SCRIPT] Document still loading (after delay), deferring actualListenerSetup for URL:', document.location.href);
            document.addEventListener('DOMContentLoaded', actualListenerSetup);
          }} else {{
            console.log('[UIT SCRIPT] Document already loaded (or interactive, after delay), calling actualListenerSetup for URL:', document.location.href);
            actualListenerSetup();
          }}
        }} else {{
          console.error('[UIT SCRIPT] FATAL: Binding still not found after delay for document:', document.location.href, '. Listeners NOT attached.');
        }}
      }}, 10);
    }}
  }}
  
  // Start the setup process
  deferredSetupCaller();

}})();
"""

    def __init__(self, *, context: Optional[Any] = None, page: Optional[Any] = None, cdp_client: Optional[Any] = None):
        """Init with a Playwright BrowserContext and an initial Page."""
        if context is None and cdp_client is not None:
            context = cdp_client 
        self.context = context
        self.page = page
        self.events: List[InputEvent] = []
        self.is_recording = False
        self.current_url: str = ""
        self._cleanup: List[Callable[[], None]] = []
        self._script_source = self._JS_TEMPLATE.format(binding=self.BINDING)
        logger.debug(f"USER_INPUT_TRACKER: Formatted _script_source (first 120 chars): {self._script_source[:120]}")
        logger.debug(f"USER_INPUT_TRACKER: Length of _script_source: {len(self._script_source)}")

    async def start_tracking(self):
        if self.is_recording:
            return True # Already recording
        if not self.page:
            logger.error("UserInputTracker: Page is not set, cannot start tracking.")
            return False
        if not self.context: # self.context here is the Playwright BrowserContext
            logger.error("UserInputTracker: Context is not set, cannot start tracking.")
            return False
            
        try:
            # Context-level binding (expose_binding, add_init_script) is now assumed to be handled 
            # by CustomBrowserContext before this UserInputTracker instance is created or started.
            # UserInputTracker will focus on page-specific listeners.

            # logger.info("Ensuring page-specific setup in UserInputTracker.start_tracking") # Optional: for debugging
            
            await self._setup_page_listeners(self.page) # Setup for the initial self.page
            
            # Listen for new pages in the context to set up their listeners
            # This is now primarily handled by CustomBrowserContext._on_binding_wrapper
            # Removing the redundant listener setup here to avoid conflicts.
            # bound_setup_page_listeners = lambda p: asyncio.create_task(self._setup_page_listeners(p))
            # self.context.on("page", bound_setup_page_listeners)
            # self._cleanup.append(lambda: self.context.remove_listener("page", bound_setup_page_listeners) if self.context else None)

            self.is_recording = True
            self.current_url = self.page.url if self.page else ""
            logger.info("User-input tracking started (listeners configured by UserInputTracker)")
            return True
        except Exception as e: # Added 'e' to log the specific exception
            logger.exception(f"Failed to start tracking in UserInputTracker: {e}")
            await self.stop_tracking() # Attempt to clean up if start fails
            return False

    async def stop_tracking(self):
        if not self.is_recording:
            return
        for fn in self._cleanup:
            try:
                fn()
            except Exception as e_cleanup:
                logger.debug(f"Error during cleanup function: {e_cleanup}")
                pass
        self._cleanup.clear()
        self.is_recording = False
        logger.info("User-input tracking stopped")

    # Renamed from _setup_page to reflect its new role
    async def _setup_page_listeners(self, page):
        """Set up page-specific listeners. Binding and init script are context-level."""
        if not page or page.is_closed():
            logger.warning(f"Attempted to set up listeners on a closed or invalid page: {page.url if page else 'N/A'}")
            return

        logger.debug(f"START: Setting up page-specific listeners for page: {page.url}")

        # 1. Playwright-level navigation listener (for NavigationEvent)
        # Use a wrapper to handle potential errors if page closes before lambda runs
        def playwright_nav_handler(frame):
            if not page.is_closed():
                self._on_playwright_nav(page, frame)
            else:
                logger.debug(f"Page closed, skipping _on_playwright_nav for url: {frame.url if frame else 'N/A'}")

        page.on("framenavigated", playwright_nav_handler)
        self._cleanup.append(lambda: page.remove_listener("framenavigated", playwright_nav_handler) if not page.is_closed() else None)

        # 2. Ensure script is evaluated on all existing frames of this page.
        #    add_init_script on context handles future frames/navigations.
        logger.debug(f"Evaluating main tracking script in all frames for page: {page.url}")
        eval_results = await self._eval_in_all_frames(page, self._script_source)
        logger.debug(f"Finished evaluating main tracking script in all frames for page: {page.url}. Results (per frame): {eval_results}")

        # 3. Listeners for dynamic frames within this specific page to re-evaluate script.
        #    (Optional, as context-level add_init_script should cover new frames, but kept for safety)
        async def safe_eval_on_frame(frame_to_eval):
            if not frame_to_eval.is_detached():
                await self._safe_eval(frame_to_eval, self._script_source)
            else:
                logger.debug("Frame detached, skipping _safe_eval.")
        
        # Store lambdas for removal
        frame_attached_lambda = lambda fr: asyncio.create_task(safe_eval_on_frame(fr))
        frame_navigated_lambda = lambda fr: asyncio.create_task(safe_eval_on_frame(fr))

        page.on("frameattached", frame_attached_lambda)
        page.on("framenavigated", frame_navigated_lambda) # For SPAs or dynamic content loading into frames
        self._cleanup.append(lambda: page.remove_listener("frameattached", frame_attached_lambda) if not page.is_closed() else None)
        self._cleanup.append(lambda: page.remove_listener("framenavigated", frame_navigated_lambda) if not page.is_closed() else None)

        logger.debug(f"END: Page-specific listeners setup for page: {page.url}")

    # JS → Python bridge
    async def _on_dom_event(self, _src, p: Dict[str, Any]):
        if not self.is_recording:
            return
        try:
            ts = p.get("ts", time.time()*1000)/1000.0
            url = p.get("url", self.current_url)
            mods = [m for m, f in (("alt",p.get("alt")),("ctrl",p.get("ctrl")),("shift",p.get("shift")),("meta",p.get("meta"))) if f]
            typ = p.get("type")
            sel = str(p.get("selector", ""))
            
            if typ == "mousedown":
                button_code = p.get("button") 
                button_name = "unknown"
                if isinstance(button_code, int): 
                    button_name = {0:"left",1:"middle",2:"right"}.get(button_code, "unknown")
                
                txt = p.get("text")
                                
                evt = MouseClickEvent(ts, url, "mouse_click", int(p.get("x",0)), int(p.get("y",0)), button_name, sel, txt, mods)
                self.events.append(evt)
                logger.info(f"🖱️ MouseClick, url='{evt.url}', button='{evt.button}'")
            elif typ == "keydown":
                evt = KeyboardEvent(ts, url, "keyboard_input", str(p.get("key")), str(p.get("code")), sel, mods)
                self.events.append(evt)
                logger.info(f"⌨️ KeyInput, url='{evt.url}', key='{evt.key}'")
        except Exception:
            logger.exception("Malformed DOM payload: %s", p)

    # --------------------------------------------------
    # Navigation via Playwright
    # --------------------------------------------------

    def _on_playwright_nav(self, page, frame):
        if not self.is_recording:
            return
        if frame.parent_frame is None:  # top‑level navigation
            url = frame.url
            if url and url not in (self.current_url, "about:blank"):
                nav = NavigationEvent(time.time(), url, "navigation", self.current_url, url)
                self.events.append(nav)
                self.current_url = url
                logger.info("🧭 Navigation recorded %s", url)

    # --------------------------------------------------
    # Frame‑eval helpers
    # --------------------------------------------------

    async def _eval_in_all_frames(self, page, script):
        results = []
        if not page or page.is_closed():
            logger.warning("eval_in_all_frames: Page is closed or None.")
            return [None]
        try:
            for frame in page.frames:
                if not frame.is_detached():
                    result = await self._safe_eval(frame, script)
                    results.append(result)
                else:
                    logger.debug(f"eval_in_all_frames: Frame {frame.url} is detached, skipping eval.")
                    results.append("detached_frame")
            return results
        except Exception as e:
            logger.error(f"Error during _eval_in_all_frames for page {page.url}: {e}", exc_info=True)
            return [f"error: {str(e)}"]

    async def _safe_eval(self, frame, script):
        try:
            result = await frame.evaluate(script)
            return result
        except Exception as e:
            logger.error(f"SAFE_EVAL: Error evaluating script in frame {frame.name} ({frame.url}): {str(e)}", exc_info=False)
            return f"eval_error: {str(e)}"

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    def export_events_to_json(self) -> str:
        return json.dumps({
            "version": "2.0",
            "timestamp": time.time(),
            "events": [asdict(e) for e in self.events],
        }, indent=2)

    def export_events_to_jsonl(self) -> str:
        lines = []
        last_ts = self.events[0].timestamp if self.events else 0
        for ev in self.events:
            dt = int((ev.timestamp - last_ts)*1000)
            last_ts = ev.timestamp
            # Create a dictionary from the event, ensuring 'timestamp' is not carried over.
            line_dict = asdict(ev)
            # TODO: which file is the reference? 
            # The problem statement implies 'event_type' should be 'type' in the output.
            # and 'to_url' should be 'to' for navigation events.
            # also, 'timestamp' is replaced by 't' (delta time).
            line_dict["type"] = line_dict.pop("event_type")
            if "timestamp" in line_dict: # should always be true based on InputEvent
                del line_dict["timestamp"]
            if line_dict["type"] == "navigation":
                if "to_url" in line_dict: # Ensure to_url exists
                    line_dict["to"] = line_dict.pop("to_url")
                if "from_url" in line_dict: # from_url is not in the example, remove
                    del line_dict["from_url"]
            # The example shows 'mods' instead of 'modifiers' for keyboard/mouse events.
            if "modifiers" in line_dict:
                line_dict["mods"] = line_dict.pop("modifiers")

            line_dict["t"] = dt
            lines.append(json.dumps(line_dict))
        return "\n".join(lines)
