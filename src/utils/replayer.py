import asyncio, json, logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Exceptions
# --------------------------------------------------

class Drift(Exception):
    """Raised when replay diverges from expected state."""
    def __init__(self, msg: str, event: Dict[str, Any] | None = None):
        super().__init__(msg)
        self.event = event

# --------------------------------------------------
# Trace loader helper
# --------------------------------------------------

def load_trace(path: str | Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

# --------------------------------------------------
# Replayer
# --------------------------------------------------

class TraceReplayer:
    BTN_MAP = {"left": "left", "middle": "middle", "right": "right"}
    MOD_MAP = {"alt": "Alt", "ctrl": "Control", "shift": "Shift", "meta": "Meta"}

    def __init__(self, page, trace: List[Dict[str, Any]]):
        self.page, self.trace = page, trace
        self._clicked_with_selector = False

    # ------------- main loop -------------

    async def play(self, speed: float = 2.0):
        i = 0
        while i < len(self.trace):
            ev = self.trace[i]
            
            # New concise and iconic log format
            log_type = ev["type"]
            current_event_url = ev.get("url", "N/A") # URL from the event itself

            log_message_elements = []

            if log_type == "mouse_click":
                log_message_elements.append("🖱️ MouseClick")
                button_text = ev.get("text")
                selector = ev.get("selector")
                
                if button_text:
                    log_message_elements.append(f"button_text:\"{button_text}\"")
                elif selector: 
                    log_message_elements.append(f"selector:\"{selector}\"")
                else: 
                    log_message_elements.append(f"xy:({ev.get('x', 'N/A')},{ev.get('y', 'N/A')})")
                
                button_type = ev.get("button", "left")
                if button_type != "left": # Only show if not default left click
                    log_message_elements.append(f"button:\"{button_type}\"")
                log_message_elements.append(f"url='{current_event_url}'")
            
            elif log_type == "keyboard_input":
                log_message_elements.append("⌨️ KeyInput")
                key_val = ev.get("key")
                log_message_elements.append(f"key:'{key_val}'")
                
                modifiers = ev.get("modifiers")
                if modifiers: # Only show if modifiers are present
                    log_message_elements.append(f"mods:{modifiers}")
                
                log_message_elements.append(f"url='{current_event_url}'")

            elif log_type == "navigation":
                log_message_elements.append("🌐 Navigation")
                to_url = ev.get("to")
                log_message_elements.append(f"to='{to_url}'")

            else: # Generic fallback for other event types like scroll, viewport_change etc.
                log_message_elements.append(f"{log_type.replace('_', ' ').title()}")
                s = ev.get("selector")
                if s: log_message_elements.append(f"selector:\"{s}\"")
                if 'x' in ev and 'y' in ev:
                     log_message_elements.append(f"coords:({ev.get('x')},{ev.get('y')})")
                log_message_elements.append(f"url='{current_event_url}'")

            logger.info(", ".join(log_message_elements))
            
            # Delay logic
            event_delay_ms = ev.get("t", 0)
            if event_delay_ms > 10: # Log only if delay is > 10ms (to avoid spamming for 0ms delays)
                 logger.debug(f"Pausing for {event_delay_ms/1000.0:.3f}s (speed adjusted: {event_delay_ms/1000.0/speed:.3f}s)")
            await asyncio.sleep(event_delay_ms / 1000.0 / speed)

            if ev["type"] == "keyboard_input":
                consumed = await self._batch_type(i)
                i += consumed
                continue

            await self._apply(ev) 
            await self._verify(ev)
            i += 1

    # ------------- batching -------------

    async def _batch_type(self, idx: int) -> int:
        ev_start_batch = self.trace[idx]
        sel, mods = ev_start_batch.get("selector"), ev_start_batch.get("modifiers", [])
        text_to_type = ""
        
        current_idx_in_trace = idx
        
        first_key = ev_start_batch.get("key", "")
        is_first_key_batchable = len(first_key) == 1 and not mods

        if is_first_key_batchable:
            text_to_type = first_key
            current_idx_in_trace = idx + 1
            while current_idx_in_trace < len(self.trace):
                nxt = self.trace[current_idx_in_trace]
                if nxt["type"] != "keyboard_input" or nxt.get("t",1) != 0: break 
                if nxt.get("selector") != sel: break
                if nxt.get("modifiers"): break
                
                next_key_char = nxt.get("key", "")
                if len(next_key_char) == 1:
                    text_to_type += next_key_char
                    current_idx_in_trace += 1
                else:
                    break
            current_idx_in_trace -= 1

        num_events_processed = 0
        if len(text_to_type) > 1:
            await self._apply_type(sel, text_to_type, [], ev_start_batch)
            await self._verify(ev_start_batch)
            num_events_processed = current_idx_in_trace - idx + 1
        else:
            await self._apply(ev_start_batch) 
            await self._verify(ev_start_batch)
            num_events_processed = 1
            
        return num_events_processed

    async def _apply_type(self, sel: Optional[str], text: str, mods: List[str], original_event_for_log: Dict[str, Any]):
        log_sel_for_type = sel or "N/A"
        logger.debug(f"APPLYING BATCH TYPE: '{text}' -> {log_sel_for_type}")

        if sel:
            try:
                await self.page.locator(sel).first.focus(timeout=800)
            except Exception as e_focus:
                logger.debug(f"Focus failed for selector '{sel}' during batch type: {e_focus.__class__.__name__}")
                pass 
        
        mapped_mods = [self.MOD_MAP[m] for m in mods if m in self.MOD_MAP]
        for m_down in mapped_mods: await self.page.keyboard.down(m_down)
        
        try:
            await self.page.keyboard.type(text)
        except Exception as e_type:
            logger.error(f"Error during page.keyboard.type('{text}'): {e_type.__class__.__name__} - {str(e_type)}")

        for m_up in reversed(mapped_mods): await self.page.keyboard.up(m_up)
        logger.debug(f"✅ done BATCH TYPE: '{text}' -> {log_sel_for_type}")

    # ------------- apply -------------

    async def _apply(self, ev: Dict[str, Any]):
        typ = ev["type"]
        sel_event = ev.get("selector")
        logger.debug(f"APPLYING ACTION: {typ} for sel={sel_event or 'N/A'}, key={ev.get('key','N/A')}")

        if typ == "navigation":
            target = ev["to"]
            if not self._url_eq(self.page.url, target):
                try:
                    await self.page.goto(target, wait_until="domcontentloaded", timeout=15000)
                except Exception as e:
                    logger.warning("goto timeout %s – continuing for %s", e.__class__.__name__, target)
            await self.page.bring_to_front()
            logger.info(f"NAVIGATED: {target}")
            return

        if typ == "mouse_click":
            btn = ev.get("button", "left")
            self._clicked_with_selector = False
            if sel_event:
                loc = await self._resolve_click_locator(sel_event)
                if loc:
                    try:
                        await loc.scroll_into_view_if_needed()
                        logger.debug(f"Waiting for element '{sel_event}' to be enabled before click.")
                        element_handle = await loc.element_handle()
                        if not element_handle:
                            raise Exception(f"Could not get element handle for selector: {sel_event}")

                        await self.page.wait_for_function(
                            expression="e => !e.disabled && (!e.hasAttribute('aria-disabled') || e.getAttribute('aria-disabled') === 'false')",
                            arg=element_handle,
                            polling='raf',
                            timeout=2000
                        )
                        logger.debug(f"Element '{sel_event}' is enabled. Proceeding with click (force=True).")
                        await loc.click(button=self.BTN_MAP.get(btn, "left"), timeout=1500, force=True)
                        self._clicked_with_selector = True
                        logger.debug(f"selector click OK → {sel_event}")
                        return
                    except Exception as e_click:
                        logger.warning(f"selector click failed for {sel_event}: {e_click.__class__.__name__} ({str(e_click)}) – fallback XY")
            
            log_x, log_y = ev.get("x"), ev.get("y")
            logger.debug(f"fallback XY click {log_x},{log_y}")
            await self.page.mouse.click(log_x or 0, log_y or 0, button=self.BTN_MAP.get(btn, "left"))
            return
        
        if typ == "keyboard_input":
            key_to_press = ev["key"]
            modifiers_for_press = ev.get("modifiers", [])
            sel_for_press = ev.get("selector")
            logger.debug(f"APPLYING SINGLE KEY PRESS: '{key_to_press}' (mods: {modifiers_for_press}) -> {sel_for_press or 'no specific target'}")

            if sel_for_press:
                try:
                    target_loc_key_press = self.page.locator(sel_for_press).first
                    if await target_loc_key_press.count() > 0:
                         await target_loc_key_press.focus(timeout=800)
                    else:
                        logger.warning(f"Target element for key press not found: {sel_for_press}")
                except Exception as e_focus_single_key:
                    logger.debug(f"Focus failed for selector '{sel_for_press}' during single key press: {e_focus_single_key.__class__.__name__}")

            mapped_mods_press = [self.MOD_MAP[m] for m in modifiers_for_press if m in self.MOD_MAP]
            for m_down_key in mapped_mods_press: await self.page.keyboard.down(m_down_key)
            
            try:
                await self.page.keyboard.press(key_to_press)
            except Exception as e_press:
                 logger.error(f"Error during page.keyboard.press('{key_to_press}'): {e_press.__class__.__name__} - {str(e_press)}")

            for m_up_key in reversed(mapped_mods_press): await self.page.keyboard.up(m_up_key)
            logger.debug(f"✅ done SINGLE KEY PRESS: '{key_to_press}' -> {sel_for_press or 'no specific target'}")
            return

        logger.debug(f"✅ done {typ} (no specific apply action in this path or already handled)")

    async def _resolve_click_locator(self, sel: str) -> Optional[Any]:
        if not sel: return None 

        initial_loc = self.page.locator(sel).first 
        if await initial_loc.count() > 0:
            return initial_loc
        
        if sel.endswith('>span') or sel.endswith('>span>span'):
            logger.debug("Selector '%s' ends with >span. Attempting to find ancestor button.", sel)
            try:
                ancestor_button_loc = initial_loc.locator('xpath=ancestor::button | ancestor::*[@role="button"]').first
                if await ancestor_button_loc.count() > 0:
                    logger.debug("Found ancestor button for '%s'. Using it for click.", sel)
                    return ancestor_button_loc
                else:
                    logger.debug("No ancestor button found for '%s'. Will try original selector if it exists.", sel)
            except Exception as e_ancestor:
                logger.debug("Error finding ancestor button for '%s': %s. Trying original selector.", sel, e_ancestor)
        return None

    # ------------- verify -------------

    async def _verify(self, ev: Dict[str, Any]):
        typ = ev["type"]

        if typ == "navigation":
            if not self._url_eq(self.page.url, ev["to"]):
                raise Drift("URL drift: expected %s, got %s" % (ev["to"], self.page.url), ev)
            return
        
        if typ == "mouse_click" and self._clicked_with_selector and ev.get("selector"):
            sel_from_event_verify = ev["selector"] 
            if "tweetButton" in sel_from_event_verify: 
                try:
                    await self.page.wait_for_selector('[data-testid="toast"]:has-text("sent")', timeout=4000)
                    logger.info("Tweet post confirmation found for: %s", sel_from_event_verify)
                except Exception:
                    logger.warning("Tweet post confirmation NOT found for: %s", sel_from_event_verify)
                    raise Drift("Tweet not posted (confirmation toast not found)", ev)
            elif ev.get("text") is not None:
                recorded_text = ev["text"]
                try:
                    verify_loc = await self._resolve_click_locator(sel_from_event_verify) 
                    if verify_loc and await verify_loc.count() > 0:
                        current_text = (await verify_loc.inner_text()).strip()
                        if current_text == recorded_text:
                            logger.info(f"Inner text matched for {sel_from_event_verify}: '{recorded_text}'")
                        else:
                            logger.warning(f"Text drift for {sel_from_event_verify}: expected '{recorded_text}', got '{current_text}'")
                            raise Drift(f"Text drift: expected '{recorded_text}', got '{current_text}'", ev)
                    else:
                        logger.warning(f"Cannot verify text for {sel_from_event_verify}, element not found by re-resolving.")
                except Exception as e_text_verify:
                    logger.warning(f"Error during text verification for {sel_from_event_verify}: {str(e_text_verify)}")
            return
        
        if typ == "keyboard_input":
            try:
                active_element_focused = await self.page.evaluate("document.activeElement !== null && document.activeElement !== document.body")
                if not active_element_focused:
                    logger.debug("No specific element has focus after typing for event: %s", ev.get("selector"))
            except Exception as e:
                logger.debug("Error checking active element after typing: %s", e)
            return

    # ---------- util ----------
    @staticmethod
    def _url_eq(a, b): 
        if not a or not b: return False
        pa, pb = urlparse(a), urlparse(b)
        if pa.netloc.replace('www.','') != pb.netloc.replace('www.',''): return False
        if pa.path.rstrip('/') != pb.path.rstrip('/'): return False
        KEEP = {'q','tbm','hl'} 
        qa = {k:v for k,v in parse_qs(pa.query).items() if k in KEEP}
        qb = {k:v for k,v in parse_qs(pb.query).items() if k in KEEP}
        return qa == qb

# --------------------------------------------------
# CLI demo (optional)
# --------------------------------------------------

async def _cli_demo(url: str, trace_path: str):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(url)
        rep = TraceReplayer(page, load_trace(trace_path))
        try:
            await rep.play(speed=3)
            print("✅ Replay completed")
        except Drift as d:
            print("⚠️  Drift:", d)
        await browser.close()

if __name__ == "__main__":
    import sys, asyncio as _a
    _a.run(_cli_demo(sys.argv[1], sys.argv[2]))