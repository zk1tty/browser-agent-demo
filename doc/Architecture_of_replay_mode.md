## User Jouenry

1. A user open a custom Chrome browser:   
    the system operates with a browser instance, which would be a custom Chrome browser in your setup.
2. The CDP session is connected between a custom Chrome browser and webui python run worker:   
    The UserInputTracker (managed by CustomBrowserContext) requires an active CDP session to listen to browser events.
3. A user click "Start Recording" button, and start browsing:   
    While the "Start Recording" button is a UI abstraction, it would trigger a call to CustomBrowserContext.start_user_input_tracking(). This, in turn, initializes and starts the UserInputTracker.
4. The event listener is on and start listening the event history:   
    UserInputTracker.start_tracking() registers the necessary CDP event listeners (e.g., for mouse clicks, key presses, navigation).
5. A user click "Stop Recording" button:   
    This UI action would map to a call to CustomBrowserContext.stop_user_input_tracking().
6. The event listener is off, and export the event history:   
    UserInputTracker.stop_tracking() unregisters the CDP listeners.
    The CustomBrowserContext.stop_user_input_tracking() method then handles saving the recorded events to a file if configured to do so.

## Event History File

1. Where to store
- The file is saved by the CustomBrowserContext.stop_user_input_tracking() method.
- The save path is determined by the tracking_save_path attribute of CustomBrowserContext.
- This path is initialized from a configuration object (BrowserContextConfig), specifically the save_input_tracking_path attribute.
- If not specified in the config, it defaults to a subdirectory ./tmp/input_tracking/ relative to where your application is running.
- The filename: input_trace_{timestamp}.json.

2. Data structure of the event history file?

- The UserInputTracker.export_events_to_json() method defines the structure. It's a JSON object with the following top-level keys:
    - "version": A version string (e.g., "1.0").
    - "timestamp": A Unix timestamp indicating when the file was exported.
    - "events": A list, where each item in the list is an object representing a single recorded user event.
- Each event object in the "events" list is a dictionary derived from one of the event dataclasses (InputEvent, MouseClickEvent, KeyboardEvent, NavigationEvent, etc.). The fields include:
    1. Common fields from InputEvent: timestamp (when the event occurred), url (current page URL), event_type (e.g., "mouse_click", "keyboard_input", "navigation").
    2. Specific fields for each event type:
MouseClickEvent: x, y, button, element_selector, element_text.
    - KeyboardEvent: key, modifiers (like Shift, Ctrl), text (the actual text typed, if any), and the recently added element_selector and element_text for the target input field.
    - NavigationEvent: from_url, to_url.
Other event types would have their respective fields.

Here's a simplified example of what the JSON structure might look like:

```
{
    "version": "1.0",
    "timestamp": 1678886400.0,
    "events": [
    {
        "timestamp": 1678886300.123,
        "url": "https://example.com/page1",
        "event_type": "navigation",
        "from_url": "https://example.com/home",
        "to_url": "https://example.com/page1"
    },
    {
        "timestamp": 1678886305.456,
        "url": "https://example.com/page1",
        "event_type": "mouse_click",
        "x": 100,
        "y": 200,
        "button": "left",
        "element_selector": "button#submit-btn",
        "element_text": "Submit"
    },
    {
        "timestamp": 1678886310.789,
        "url": "https://example.com/page1",
        "event_type": "keyboard_input",
        "key": "H",
        "modifiers": ["shift"],
        "text": "H",
        "element_selector": "input#username",
        "element_text": ""
    }
    // ... more events
    ]
}
```