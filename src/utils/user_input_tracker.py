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

    # one canonical binding name used in JS <-> Python bridge
    BINDING = "__uit_relay"

    # JS snippet; doubled braces survive str.format.
    _JS_TEMPLATE = """
    (function() {{
        const binding = "{binding}";
        const send = (type, e) => {{
            // dev‑time feedback in DevTools
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
            }} else {{
                console.warn('[UIT] binding', binding, 'missing on window');
            }}
        }};
        document.addEventListener('mousedown', e => send('mousedown', e), true);
        document.addEventListener('keydown',   e => send('keydown',   e), true);
        console.log('[UIT] listeners ready');
    }})();
    """

    def __init__(self, *, cdp_client: Optional[Any] = None, page: Optional[Any] = None):
        self.cdp_client = cdp_client
        self.page = page
        self.events: List[InputEvent] = []
        self._cleanup: List[Callable[[], None]] = []
        self.is_recording = False
        self.current_url = ""

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    async def start_tracking(self, *, cdp_client: Optional[Any] = None, page: Optional[Any] = None) -> bool:
        if cdp_client:
            self.cdp_client = cdp_client
        if page:
            self.page = page
        if not (self.cdp_client and self.page):
            logger.error("UserInputTracker requires CDP client and Playwright page")
            return False
        if self.is_recording:
            return True
        try:
            await self._register()
            await self.page.bring_to_front()  # ensure focus for key events
            self.is_recording = True
            self.current_url = self.page.url
            logger.info("User‑input tracking started")
            return True
        except Exception:
            logger.exception("Failed to start tracking")
            await self._unregister()
            return False

    async def stop_tracking(self):
        if not self.is_recording:
            return
        await self._unregister()
        self.is_recording = False
        logger.info("User‑input tracking stopped")

    # --------------------------------------------------
    # Internal registration / teardown
    # --------------------------------------------------

    async def _register(self):
        # 1. enable Page domain for navigation events
        await self.cdp_client.send("Page.enable")
        self.cdp_client.on("Page.frameNavigated", self._on_cdp_navigation)
        self._cleanup.append(lambda: self.cdp_client.off("Page.frameNavigated", self._on_cdp_navigation))

        # 2. expose python binding first
        await self.page.expose_binding(self.BINDING, self._on_dom_event)

        # 3. build & persist init script (future navs)
        script = self._JS_TEMPLATE.format(binding=self.BINDING)
        await self.page.add_init_script(script)

        # 4. run now in current frames
        await self._eval_in_all_frames(script)

        # 5. keep new / navigated frames covered
        self.page.on("frameattached",  lambda f: asyncio.create_task(self._safe_eval(f, script)))
        self.page.on("framenavigated", lambda f: asyncio.create_task(self._safe_eval(f, script)))
        self._cleanup.append(lambda: self.page.off("frameattached", None))
        self._cleanup.append(lambda: self.page.off("framenavigated", None))

    async def _unregister(self):
        for fn in self._cleanup:
            try:
                fn()
            except Exception:
                pass
        self._cleanup.clear()

    # --------------------------------------------------
    # Helpers for frame eval
    # --------------------------------------------------

    async def _eval_in_all_frames(self, script: str):
        await self._safe_eval(self.page.main_frame, script)
        for fr in self.page.frames:
            await self._safe_eval(fr, script)

    async def _safe_eval(self, frame, script: str):
        try:
            await frame.evaluate(script)
        except Exception:
            pass  # cross‑origin or sandboxed frames may refuse

    # --------------------------------------------------
    # Bridge: JS → Python
    # --------------------------------------------------

    async def _on_dom_event(self, _source, p: Dict[str, Any]):
        if not self.is_recording:
            return
        try:
            ts = p.get("ts", time.time()*1000)/1000.0
            url = p.get("url", self.current_url)
            mods = [m for m, f in (("alt",p.get("alt")), ("ctrl",p.get("ctrl")), ("shift",p.get("shift")), ("meta",p.get("meta"))) if f]
            typ = p.get("type")
            if typ == "mousedown":
                btn_map = {0:"left", 1:"middle", 2:"right"}
                button  = btn_map.get(p.get("button"), "unknown")
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
    # CDP navigation handler
    # --------------------------------------------------

    def _on_cdp_navigation(self, ev: Dict[str, Any]):
        if not self.is_recording:
            return
        url = ev.get("frame", {}).get("url")
        if url and url not in (self.current_url, "about:blank"):
            nav = NavigationEvent(time.time(), url, "navigation", self.current_url, url)
            self.events.append(nav)
            self.current_url = url

    # --------------------------------------------------
    # Export helpers
    # --------------------------------------------------

    def export_events_to_json(self) -> str:
        return json.dumps({
            "version": "1.5",
            "timestamp": time.time(),
            "events": [asdict(e) for e in self.events],
        }, indent=2)

    # --------------------------------------------------
    # Synchronous wrappers (optional)
    # --------------------------------------------------

    def start_sync(self, **kw):
        return asyncio.get_event_loop().run_until_complete(self.start_tracking(**kw))

    def stop_sync(self):
        return asyncio.get_event_loop().run_until_complete(self.stop_tracking())
