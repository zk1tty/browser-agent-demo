import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Exceptions
# --------------------------------------------------

class Drift(Exception):
    """Raised when deterministic replay diverges from expected page state."""
    def __init__(self, message: str, event: Dict[str, Any] | None = None):
        super().__init__(message)
        self.event = event

# --------------------------------------------------
# Trace loader helper
# --------------------------------------------------

def load_trace(path: str | Path) -> List[Dict[str, Any]]:
    """Read a .jsonl trace file produced by UserInputTracker."""
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events

# --------------------------------------------------
# Main replayer
# --------------------------------------------------

class TraceReplayer:
    """Deterministically replays a trace; raises Drift on mismatch."""

    BTN_MAP = {"left": "left", "middle": "middle", "right": "right"}
    MOD_MAP = {"alt": "Alt", "ctrl": "Control", "shift": "Shift", "meta": "Meta"}

    def __init__(self, page, trace: List[Dict[str, Any]]):
        self.page = page
        self.trace = trace

    # ------------- public -------------

    async def play(self, speed: float = 2.0):
        """Iterate through the trace; speed>1 accelerates playback."""
        for ev in self.trace:
            await asyncio.sleep(ev.get("t", 0) / 1000 / speed)
            try:
                await self._apply(ev)
                await self._verify_next_state(ev)
            except Drift:
                raise  # bubble up to agent
            except Exception as e:
                logger.exception("Unhandled error during replay; treating as drift")
                raise Drift(str(e), ev) from e

    # ------------- internals -------------

    async def _apply(self, ev: Dict[str, Any]):
        etype = ev.get("type")
        if etype == "navigation":
            await self.page.goto(ev["to"], wait_until="networkidle")
        elif etype == "mouse_click":
            sel = ev.get("selector", "")
            btn = ev.get("button", "left")
            if sel:
                try:
                    await self.page.locator(sel).first.click(button=self.BTN_MAP.get(btn, "left"), timeout=2000)
                    return
                except Exception:
                    logger.debug("Selector click failed; falling back to coordinates")
            # fallback coordinates
            await self.page.mouse.click(ev.get("x", 0), ev.get("y", 0), button=self.BTN_MAP.get(btn, "left"))
        elif etype == "keyboard_input":
            mods = [self.MOD_MAP[m] for m in ev.get("modifiers", []) if m in self.MOD_MAP]
            for m in mods:
                await self.page.keyboard.down(m)
            await self.page.keyboard.press(ev.get("key", ""))
            for m in reversed(mods):
                await self.page.keyboard.up(m)
        else:
            logger.debug("Unknown event type %s – skipping", etype)

    async def _verify_next_state(self, ev: Dict[str, Any]):
        etype = ev.get("type")
        if etype == "navigation":
            expected = ev.get("to")
            actual = self.page.url
            # --- smart comparison -------------------------------------------------
            from urllib.parse import urlparse, parse_qs
            exp, act = urlparse(expected), urlparse(actual)
            # host & path must match exactly (ignore www vs no‑www for Google)
            if exp.netloc.replace("www.","") != act.netloc.replace("www.","") or exp.path != act.path:
                raise Drift(f"URL host/path mismatch: {actual} ≠ {expected}", ev)
            # compare stable query keys (q, tbm, etc.) – ignore gs_lp, ved, iflsig …
            KEEP = {"q", "tbm", "hl"}
            exp_q = {k:v for k,v in parse_qs(exp.query).items() if k in KEEP}
            act_q = {k:v for k,v in parse_qs(act.query).items() if k in KEEP}
            if exp_q != act_q:
                raise Drift(f"URL query mismatch: {act_q} ≠ {exp_q}", ev)
            return  # navigation ok
        elif etype == "mouse_click":
            sel = ev.get("selector", "")
            btn = ev.get("button", "left")
            if sel and btn == "left":  # only assert for primary clicks with selector
                try:
                    visible = await self.page.locator(sel).is_visible(timeout=1000)
                except Exception:
                    visible = False
                if not visible:
                    raise Drift("Clicked element no longer visible", ev)
            return
        elif etype == "keyboard_input":
            try:
                active = await self.page.evaluate("document.activeElement !== null")
            except Exception:
                active = True
            if not active:
                raise Drift("No active element after key press", ev)
            return

# --------------------------------------------------
# Convenience CLI entry (optional)
# --------------------------------------------------

async def _cli_demo(url: str, trace_file: str):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(url)
        trace = load_trace(trace_file)
        rep = TraceReplayer(page, trace)
        try:
            await rep.play(speed=3.0)
            print("Replay completed without drift ✨")
        except Drift as d:
            print("Drift detected →", d)
        await browser.close()

if __name__ == "__main__":
    import sys, asyncio as _a
    _a.run(_cli_demo(sys.argv[1], sys.argv[2]))