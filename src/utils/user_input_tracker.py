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
    modifiers: List[str] = field(default_factory=list)

@dataclass
class KeyboardEvent(InputEvent):
    key: str
    code: str
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
    (function() {{
        const binding = \"{binding}\";
        const send = (type, e) => {{
            console.log('[UIT]', type, e.key ?? e.button, e.clientX ?? '', e.clientY ?? '');
            if (window[binding]) {{
                console.log('[UIT] calling python binding', binding);
                window[binding]({{
                    type,
                    ts: Date.now(),
                    url: location.href,
                    x:  e.clientX ?? null,
                    y:  e.clientY ?? null,
                    button: e.button ?? null,
                    key:    e.key    ?? null,
                    code:   e.code   ?? null,
                    alt:  e.altKey,
                    ctrl: e.ctrlKey,
                    shift:e.shiftKey,
                    meta: e.metaKey
                }});
            }}
        }};
        document.addEventListener('mousedown', e => send('mousedown', e), true);
        document.addEventListener('keydown',   e => send('keydown',   e), true);
        console.log('[UIT] listeners ready');
    }})();
    """

    def __init__(self, *, context: Optional[Any] = None, page: Optional[Any] = None, cdp_client: Optional[Any] = None):
        """Init with a Playwright BrowserContext and an initial Page."""
        # accept both `context` and (legacy) `cdp_client` for backward compatibility
        if context is None and cdp_client is not None:
            # legacy call pattern: tracker(context=..., page=..., cdp_client=ctx)
            context = cdp_client  # treat provided CDP client as BrowserContext
        self.context = context
        self.page = page
        self.events: List[InputEvent] = []
        self.is_recording = False
        self.current_url: str = ""
        self._cleanup: List[Callable[[], None]] = []
        # we keep one compiled script ready
        self._script_source = self._JS_TEMPLATE.format(binding=self.BINDING)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    async def start_tracking(self):
        if self.is_recording:
            return True
        try:
            await self._setup_page(self.page)            # existing page
            self.context.on("page", lambda p: asyncio.create_task(self._setup_page(p)))
            self._cleanup.append(lambda: self.context.off("page", None))
            self.is_recording = True
            self.current_url = self.page.url
            logger.info("User‑input tracking started")
            return True
        except Exception:
            logger.exception("Failed to start tracking")
            await self.stop_tracking()
            return False

    async def stop_tracking(self):
        if not self.is_recording:
            return
        for fn in self._cleanup:
            try:
                fn()
            except Exception:
                pass
        self._cleanup.clear()
        self.is_recording = False
        logger.info("User‑input tracking stopped")

    # --------------------------------------------------
    # Per‑page setup
    # --------------------------------------------------

    async def _setup_page(self, page):
        """Expose binding, inject script, handle nav + new frames on one page."""
        # 1. binding
        await page.expose_binding(self.BINDING, self._on_dom_event)

        # 2. navigation listener (Playwright-level)
        page.on("framenavigated", lambda fr: self._on_playwright_nav(page, fr))
        self._cleanup.append(lambda: page.off("framenavigated", None))

        # 3. keep script on future navs and run now
        await page.add_init_script(self._script_source)
        await self._eval_in_all_frames(page, self._script_source)

        # 4. cover dynamic frames in this page
        page.on("frameattached", lambda fr: asyncio.create_task(self._safe_eval(fr, self._script_source)))
        page.on("framenavigated", lambda fr: asyncio.create_task(self._safe_eval(fr, self._script_source)))
        self._cleanup.append(lambda: page.off("frameattached", None))

    # --------------------------------------------------
    # JS → Python bridge
    # --------------------------------------------------

    async def _on_dom_event(self, _source, p: Dict[str, Any]):
        if not self.is_recording:
            return
        try:
            ts = p.get("ts", time.time()*1000)/1000.0
            url = p.get("url", self.current_url)
            mods = [m for m, f in (("alt",p.get("alt")),("ctrl",p.get("ctrl")),("shift",p.get("shift")),("meta",p.get("meta"))) if f]
            typ = p.get("type")
            if typ == "mousedown":
                button = {0:"left",1:"middle",2:"right"}.get(p.get("button"), "unknown")
                evt = MouseClickEvent(ts, url, "mouse_click", int(p.get("x",0)), int(p.get("y",0)), button, mods)
            elif typ == "keydown":
                evt = KeyboardEvent(ts, url, "keyboard_input", str(p.get("key")), str(p.get("code")), mods)
            else:
                return
            self.events.append(evt)
            logger.info("🟢 Python binding hit – recorded %s", evt)
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
        await self._safe_eval(page.main_frame, script)
        for fr in page.frames:
            await self._safe_eval(fr, script)

    async def _safe_eval(self, frame, script):
        try:
            await frame.evaluate(script)
        except Exception:
            pass

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    def export_events_to_json(self) -> str:
        return json.dumps({
            "version": "2.0",
            "timestamp": time.time(),
            "events": [asdict(e) for e in self.events],
        }, indent=2)
