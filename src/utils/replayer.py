import json, asyncio
from browser_use.browser.browser import Browser # Adjusted import for Browser
from playwright.async_api import Page # Standard Playwright import for Page
from typing import Literal

# Ensure BTN_MAP values conform to Playwright's expected literals
ButtonLiteral = Literal['left', 'middle', 'right']
BTN_MAP: dict[str, ButtonLiteral] = {
    "left": "left",
    "middle": "middle",
    "right": "right"
}

class TraceReplayer:
    def __init__(self, browser: Browser, page: Page):
        self.browser, self.page = browser, page

    async def play(self, path: str, speed: float = 1.0):
        with open(path) as f:
            events = [json.loads(l) for l in f if l.strip()]
        for ev in events:
            await asyncio.sleep(ev["t"] / 1000 / speed)
            if ev["type"] == "navigation":
                await self.page.goto(ev["to"])
            elif ev["type"] == "mouse_click":
                button_event_val = ev.get("button")
                # Default to 'left' if button is not specified or not in BTN_MAP
                button_to_press: ButtonLiteral = BTN_MAP.get(button_event_val, "left") 
                await self.page.mouse.move(ev["x"], ev["y"])
                await self.page.mouse.click(ev["x"], ev["y"],
                                            button=button_to_press)
            elif ev["type"] == "keyboard_input":
                # Map from trace format (e.g. "ctrl") to Playwright's format (e.g. "Control")
                mods_map = {"alt": "Alt", "ctrl": "Control",
                            "shift": "Shift", "meta": "Meta"}
                pressed_modifiers = []
                # ev.get("mods", []) will get the list of modifiers like ["shift", "ctrl"]
                for mod_key in ev.get("mods", []):
                    if mod_key in mods_map:
                        pressed_modifiers.append(mods_map[mod_key])
                
                for m in pressed_modifiers: await self.page.keyboard.down(m)
                await self.page.keyboard.press(ev["key"])
                for m in reversed(pressed_modifiers): await self.page.keyboard.up(m) 