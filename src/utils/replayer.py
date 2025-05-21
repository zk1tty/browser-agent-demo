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
        self._clicked_dispatch = False

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
                element_to_fill = self.page.locator(sel).first
                await element_to_fill.wait_for(state='visible', timeout=5000)
                await element_to_fill.focus(timeout=1000)
                await asyncio.sleep(0.2) # Short delay after focus before filling
                await element_to_fill.fill(text)
            except Exception as e_fill:
                logger.error(f"Error during locator.fill('{text}') for selector '{sel}': {e_fill.__class__.__name__} - {str(e_fill)}. Falling back to keyboard.type.")
                # Fallback to original keyboard.type if fill fails for some reason
                mapped_mods = [self.MOD_MAP[m] for m in mods if m in self.MOD_MAP]
                for m_down in mapped_mods: await self.page.keyboard.down(m_down)
                try:
                    await self.page.keyboard.type(text)
                except Exception as e_type:
                    logger.error(f"Error during fallback page.keyboard.type('{text}'): {e_type.__class__.__name__} - {str(e_type)}")
                for m_up in reversed(mapped_mods): await self.page.keyboard.up(m_up)
        else:
            # If no selector, fallback to general keyboard typing (less common for batched text)
            logger.warning(f"Attempting to batch type '{text}' without a selector. Using page.keyboard.type().")
            mapped_mods = [self.MOD_MAP[m] for m in mods if m in self.MOD_MAP]
            for m_down in mapped_mods: await self.page.keyboard.down(m_down)
            try:
                await self.page.keyboard.type(text)
            except Exception as e_type:
                logger.error(f"Error during page.keyboard.type('{text}') without selector: {e_type.__class__.__name__} - {str(e_type)}")
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
            # Enhanced wait after navigation
            try:
                logger.debug(f"Waiting for 'load' state after navigating to {target}")
                await self.page.wait_for_load_state('load', timeout=10000) # Wait for basic load
                logger.debug(f"'load' state confirmed for {target}. Now waiting for networkidle.")
                await self.page.wait_for_load_state('networkidle', timeout=3000) # Shorter networkidle (e.g., 3 seconds)
                await asyncio.sleep(0.3) # Small buffer
                logger.debug(f"Network idle (or timeout) confirmed for {target}")
            except Exception as e_wait:
                logger.warning(f"Timeout or error during page load/networkidle wait on {target}: {e_wait.__class__.__name__} - {str(e_wait)}")

            logger.info(f"NAVIGATED: {target}")
            return

        if typ == "mouse_click":
            btn = ev.get("button", "left")
            self._clicked_with_selector = False
            self._clicked_dispatch = False
            if sel_event:
                loc = await self._resolve_click_locator(sel_event)
                if loc:
                    element_handle = None
                    try:
                        logger.debug(f"Attempting to click resolved locator for original selector: {sel_event}")
                        
                        await loc.wait_for(state='visible', timeout=3000)
                        await loc.scroll_into_view_if_needed(timeout=3000)
                        logger.debug(f"Element '{sel_event}' (resolved to button) is visible. Attempting standard click.")
                        
                        await loc.click(button=self.BTN_MAP.get(btn, "left"), timeout=3000, delay=100)
                        self._clicked_with_selector = True
                        logger.info(f"Standard Playwright click successful for resolved locator from selector: {sel_event}")
                        await asyncio.sleep(0.25)
                        return

                    except Exception as e_click_attempt1:
                        logger.warning(f"Standard Playwright click (attempt 1) for resolved locator from '{sel_event}' failed: {e_click_attempt1.__class__.__name__} ({str(e_click_attempt1)})")
                        
                        try:
                            logger.info(f"Fallback: Attempting to dispatch click event for resolved locator from '{sel_event}'")
                            if await loc.count() > 0:
                                element_handle = await loc.element_handle(timeout=1000)
                                if element_handle:
                                    await element_handle.dispatch_event('click')
                                    self._clicked_dispatch = True
                                    self._clicked_with_selector = True
                                    logger.info(f"DispatchEvent (via element_handle) click successful for '{sel_event}'")
                                    await asyncio.sleep(0.25)
                                    return
                                else:
                                    await loc.dispatch_event('click')
                                    self._clicked_dispatch = True
                                    self._clicked_with_selector = True
                                    logger.info(f"DispatchEvent (via locator) click successful for '{sel_event}'")
                                    await asyncio.sleep(0.25)
                                    return
                            else:
                                logger.error(f"Cannot dispatch click for '{sel_event}', resolved locator is empty.")

                        except Exception as e_dispatch:
                            logger.warning(f"DispatchEvent click failed for '{sel_event}': {e_dispatch.__class__.__name__} ({str(e_dispatch)}). Falling back to XY if available.")
            
            # Fallback to XY click if selector-based attempts failed or no selector
            if not self._clicked_with_selector:
                log_x, log_y = ev.get("x"), ev.get("y")
                if log_x is not None and log_y is not None:
                    logger.info(f"Fallback: Performing coordinate-based click at ({log_x},{log_y})")
                    await self.page.mouse.click(log_x, log_y, button=self.BTN_MAP.get(btn, "left"))
                    await asyncio.sleep(0.25)
                else:
                    if sel_event:
                         logger.error(f"All click attempts failed for selector '{sel_event}' and no XY coordinates available.")
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

        # Initial locator based on the selector from the trace
        initial_loc = self.page.locator(sel).first

        # Check if the initial locator itself is a button or has role="button"
        # Use a try-catch for evaluate as the element might not exist or be stale
        try:
            if await initial_loc.count() > 0: # Ensure element exists before evaluation
                # Check if the element itself is a button or has role="button"
                is_button_or_has_role = await initial_loc.evaluate(
                    "el => el.tagName === 'BUTTON' || el.getAttribute('role') === 'button'"
                )
                if is_button_or_has_role:
                    logger.debug(f"_resolve_click_locator: Initial selector '{sel}' is already a button or has role='button'. Using it.")
                    return initial_loc
            else:
                logger.debug(f"_resolve_click_locator: Initial selector '{sel}' did not yield any elements. Will try to find ancestor.")
                # If initial_loc.count() is 0, initial_loc might not be suitable for ancestor search directly, 
                # but Playwright handles this by searching from the page if the locator is empty.
                # However, it's cleaner to ensure we have a starting point if we intend to find an ancestor *of something*.
                # For now, we will proceed, and if initial_loc is empty, the ancestor search becomes a page-wide search for a button.

        except Exception as e_eval_initial:
            logger.debug(f"_resolve_click_locator: Error evaluating initial selector '{sel}': {e_eval_initial}. Will try to find ancestor.")

        # If not, or if initial check failed, try to find an ancestor that is a button or has role="button"
        # This also covers cases where `sel` might point to an inner element of a button (e.g., a span).
        # The XPath searches for an ancestor OR self that is a button or has the role.
        # Using a more specific XPath to find the closest ancestor or self that is a button:
        # xpath=ancestor-or-self::button | ancestor-or-self::*[@role='button']
        # Playwright's loc.locator("xpath=...") will find the first such element from the perspective of `loc`.
        # If initial_loc was empty, this effectively searches from page root.
        
        # Let's try a slightly different approach for finding the button: use Playwright's :nth-match with a broader internal selector.
        # This attempts to find the *actual element* matching 'sel', then looks upwards or at itself for a button.
        # This is more robust if 'sel' is very specific to an inner element.

        # Re-fetch the initial locator to ensure we are working from the element pointed to by `sel`
        # This is important if `sel` is like 'div > span' - we want the span, then find its button parent.
        # If initial_loc.count() was 0 above, this will still be an empty locator.
        element_loc = self.page.locator(sel).first 

        if await element_loc.count() > 0:
            # Try to find a button by looking at the element itself or its ancestors
            # This combines checking self and then ascending.
            # The XPath 'ancestor-or-self::button | ancestor-or-self::*[@role="button"]' correctly finds the button.
            # We then take the .first of these, as Playwright will return them in document order (ancestors first).
            # To get the *closest* (most specific) button, we might need to be careful.
            # However, Playwright's .locator on an existing locator usually chains correctly.

            # Let's try to find the *specific* element by `sel` and then chain to find its button ancestor or self.
            # This is more reliable than a broad page search if `sel` is specific.
            potential_button_loc = element_loc.locator("xpath=ancestor-or-self::button | ancestor-or-self::*[@role='button']").first
            if await potential_button_loc.count() > 0:
                logger.debug(f"_resolve_click_locator: Found button/role=button for '{sel}' via ancestor-or-self. Using it.")
                return potential_button_loc
            else:
                logger.debug(f"_resolve_click_locator: No button ancestor found for specific element of '{sel}'. Falling back to initial locator if it exists.")
                return element_loc if await element_loc.count() > 0 else None # Fallback to the original if it existed, else None
        else:
            # If the original selector `sel` finds nothing, try a page-wide search for a button that might contain the text from `sel` if `sel` was text-based
            # This part is tricky and heuristic. For now, if `sel` finds nothing, we return None.
            logger.debug(f"_resolve_click_locator: Initial selector '{sel}' found no elements. Cannot resolve to a button.")
            return None

    # ------------- verify -------------

    async def _verify_tweet_posted(self):
        try:
            await self.page.wait_for_selector('[role=alert]:text("sent")', timeout=3000)
            logger.info("Tweet post verification successful: 'sent' toast found.")
        except Exception as e_toast:
            logger.error(f"Tweet post verification failed: 'sent' toast not found within timeout. Error: {e_toast.__class__.__name__}")
            raise Drift("Tweet not posted (confirmation toast not found after click)")

    async def _verify(self, ev: Dict[str, Any]):
        typ = ev["type"]
        sel_from_event_verify = ev.get("selector")

        if typ == "navigation":
            if not self._url_eq(self.page.url, ev["to"]):
                current_event_expected_url = ev["url"]
                nav_target_url = ev["to"]
                actual_page_url = self.page.url

                if self._url_eq(actual_page_url, nav_target_url):
                    logger.debug(f"Navigation URL verified: Expected target {nav_target_url}, Got {actual_page_url}")
                    return

                logger.warning(f"Potential Navigation URL drift: Expected target {nav_target_url}, but current URL is {actual_page_url}. Original event recorded at {current_event_expected_url}")
                
                current_event_index = -1
                try:
                    current_event_index = self.trace.index(ev)
                except ValueError:
                    logger.error("Critical: Could not find current navigation event in trace for drift recovery. Raising drift based on target mismatch.")
                    raise Drift(f"URL drift for navigation: expected target {nav_target_url}, got {actual_page_url}", ev)

                if 0 <= current_event_index < len(self.trace) - 1:
                    next_event = self.trace[current_event_index + 1]
                    logger.debug(f"Drift check for navigation: Next event is type '{next_event.get('type')}', URL '{next_event.get('url')}', To '{next_event.get('to')}'")
                    
                    if next_event.get("type") == "navigation":
                        next_event_nav_target_url = next_event.get("to")
                        next_event_recorded_at_url = next_event.get("url")

                        if next_event_nav_target_url and self._url_eq(actual_page_url, next_event_nav_target_url):
                            logger.info(f"Drift recovery for navigation: Actual URL {actual_page_url} matches TARGET of NEXT navigation. Allowing.")
                            return
                        if next_event_recorded_at_url and self._url_eq(actual_page_url, next_event_recorded_at_url):
                            logger.info(f"Drift recovery for navigation: Actual URL {actual_page_url} matches RECORDED URL of NEXT navigation. Allowing.")
                            return
                
                logger.error(f"URL drift CONFIRMED for navigation: expected target {nav_target_url}, got {actual_page_url}")
                raise Drift(f"URL drift for navigation: expected target {nav_target_url}, got {actual_page_url}", ev)
            return
        
        if typ == "mouse_click" and self._clicked_with_selector and sel_from_event_verify:
            if "tweetButton" in sel_from_event_verify:
                await self._verify_tweet_posted()
                return

            if getattr(self, "_clicked_dispatch", False):
                logger.info(f"Verification for selector '{sel_from_event_verify}': Skipped standard DOM check as dispatchEvent was used (element might be detached/changed).")
                return
            
            recorded_text = ev.get("text")
            if recorded_text is not None:
                try:
                    verify_loc = await self._resolve_click_locator(sel_from_event_verify) 
                    if verify_loc and await verify_loc.count() > 0:
                        current_text = (await verify_loc.inner_text(timeout=1000)).strip()
                        if current_text == recorded_text:
                            logger.info(f"Inner text matched for {sel_from_event_verify}: '{recorded_text}'")
                        else:
                            logger.warning(f"Text drift for {sel_from_event_verify}: expected '{recorded_text}', got '{current_text}'")
                    else:
                        logger.warning(f"Cannot verify text for {sel_from_event_verify}, element not found by re-resolving after click.")
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

        # Selector verification (if applicable)
        # This part remains unchanged from your existing logic if you have it.
        # For example, if a click was supposed to happen on a selector:
        if ev["type"] == "mouse_click" and ev.get("selector") and not self._clicked_with_selector:
            # This implies the fallback XY click was used, which can be a form of drift.
            # You might want to log this or handle it as a minor drift.
            logger.debug(f"Verification: Click for selector '{ev['selector']}' used XY fallback.")

        # URL drift check
        current_event_expected_url = ev["url"]
        actual_page_url = self.page.url

        if not self._url_eq(actual_page_url, current_event_expected_url):
            logger.warning(f"Potential URL drift: expected {current_event_expected_url} (from event record), got {actual_page_url} (actual browser URL).")

            current_event_index = -1
            try:
                # Find the index of the current event 'ev' in self.trace
                # This is okay for moderately sized traces. Consider passing index if performance becomes an issue.
                current_event_index = self.trace.index(ev)
            except ValueError:
                logger.error("Critical: Could not find current event in trace for drift recovery. This shouldn't happen. Raising original drift.")
                raise Drift(f"URL drift (and event indexing error): expected {current_event_expected_url}, got {actual_page_url}", ev)

            if 0 <= current_event_index < len(self.trace) - 1:
                next_event = self.trace[current_event_index + 1]
                logger.debug(f"Drift check: Next event is type '{next_event.get('type')}', URL '{next_event.get('url')}', To '{next_event.get('to')}'")
                
                if next_event.get("type") == "navigation":
                    next_event_target_url = next_event.get("to")
                    next_event_recorded_at_url = next_event.get("url")

                    # Condition 1: The browser is AT the target URL of the NEXT navigation event.
                    # This means the current navigation (ev) effectively led to where next_event will go.
                    if next_event_target_url and self._url_eq(actual_page_url, next_event_target_url):
                        logger.info(f"Drift recovery: Actual URL {actual_page_url} matches TARGET ('to') of the NEXT navigation event. Allowing.")
                        return

                    # Condition 2: The browser is AT the URL where the NEXT navigation event was RECORDED.
                    # This means the current navigation (ev) might have been part of a quick redirect chain,
                    # and the page has landed on the 'url' from which the next_event was initiated.
                    # This is relevant if next_event_target_url is different from next_event_recorded_at_url
                    if next_event_recorded_at_url and self._url_eq(actual_page_url, next_event_recorded_at_url):
                        logger.info(f"Drift recovery: Actual URL {actual_page_url} matches RECORDED URL ('url') of the NEXT navigation event. Allowing.")
                        return
            
            # If no recovery condition met, raise the original drift error
            logger.error(f"URL drift CONFIRMED after checks: expected {current_event_expected_url} (from event record), got {actual_page_url} (actual browser URL).")
            raise Drift(f"URL drift: expected {current_event_expected_url}, got {actual_page_url}", ev)
        else:
            logger.debug(f"URL verified: Expected {current_event_expected_url}, Got {actual_page_url}")

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