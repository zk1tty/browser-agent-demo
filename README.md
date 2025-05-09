[<img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-title.png" alt="Rebrowse Title" width="full"/>](https://rebrowse.me)

<br/>

[![n0rizkitty](https://img.shields.io/twitter/follow/n0rizkitty?style=social)](https://x.com/n0rizkitty)

## About Rebrowse

Rebrowse is a powerful tool that converts screen recordings into automated workflow agents.   

Key features of Rebrowse:
- Record your screen once and automate the workflow forever
- Transform recordings into shareable workflow agents
- Support for cross-application workflows
- Easy sharing and collaboration   

Explore our Showcases at [rebrowse.me](https://rebrowse.me)

## Watch Demo

- Scenario: Book an Airbnb apartment with specific conditions
- Environment: Chrome browser with a personal Google account

[![Airbnb Booking Demo](https://img.youtube.com/vi/1kQu8oYG-2g/0.jpg)](https://youtu.be/1kQu8oYG-2g)

*Click the thumbnail to watch [on YouTube](https://youtu.be/1kQu8oYG-2g)!*

Follow [me on Twitter](https://x.com/n0rizkitty) for the latest updates!

## Roadmap

We're gonna build a marketplace, where users can share cross-app workflows by recording, instead of node editors like Zapier or n8n.      
Check out our [Roadmap](./ROADMAP.md).   

<img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-flywheel.png" alt="Rebrowse Flywheel" width="350" style="display: block; margin-left: 0;">

---
## Installation Guide

<img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-setting.png" alt="Rebrowse Setting" width="full" />

### Prerequisites
- Python 3.8 or higher
- Git
- Chrome browser (for browser automation)
- Gradio (for the web interface)

### Step 1: Clone the Repository
```bash
git clone https://github.com/zk1tty/rebrowse-app.git
cd rebrowse-app
```

### Step 2: Set Up Python Environment

- We recomended to use uv as reccomnede at the [brwoser-use doc](https://docs.browser-use.com/quickstart).
- You can download uv from [here](https://docs.astral.sh/uv/#installation)

```bash
# Create and activate a virtual environment
# We recomended to use uv.
uv venv --python 3.11

# For Mac/Linux:
source .venv/bin/activate

# For Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
1. Copy `.env.examlpe` file to `.env` file in the project root.

2. Add API key, and update chrome path
    ```bash
    # LLM API Keys
    OPENAI_API_KEY=your_openai_api_key
    ```

    ```bash
    # Chrome Configuration
    ## TODO: replace {username} with your OS username
    CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    CHROME_USER_DATA="/Users/username/Library/Application Support/Google/Chrome"
    CHROME_CDP="http://localhost:9222"
    ```
### Step 5: Close all opening Chrome tabs!! IMPORTANT!!

<img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/step_quit_chrome.png" alt="Quit Chrome" width="350" style="display: block; margin-left: 0;">

Or, run this command.

```
pkill -9 "Google Chrome"
```


### Step 6: Launch the web Application
```bash
# Start with Gradio
gradio webui.py

# Or run directly with Python
python webui.py
```
The application will be available at `http://127.0.0.1:7860`

### Step 7: Use FireFox or Safari to open 127.0.0.1:7860
- DO NOT use Chrome to open this web app.
- Open Safari or FIrefox, and go to "http://127.0.0.1:7860"


### Step 8: Run a clean Chrome with CDP session
To enable the agent to use your personal Chrome browser, you need to start Chrome with specific flags:

1. First, identify your Chrome application path:
   - macOS: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`
   - Windows: `C:\Program Files\Google\Chrome\Application\chrome.exe`
   - Linux: `/usr/bin/google-chrome`

2. Execute Chrome with the following flags:
    ⚠️ Replace {username} with your OS username
    ```bash
    # macOS example
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --remote-debugging-port=9222 \
    --user-data-dir="/Users/{username}/Library/Application\ Support/Google/Chrome" \
    --profile-directory="Default" \
    --no-first-run \
    --no-default-browser-check
    ```
3. Make sure that devtool websocket session is available
    ```
    DevTools listening on ws://127.0.0.1:9222/devtools/browser/1bbd94a9-aed9-4462-bc25-fddec9d9663c
    ```
### Step 9: Ready to play ^^
- Go to "Choose Agent" tab
- PItck up Preset Task
- Try "Run Agent"
- Enjoy your browseing agent.

---
## Background

This project builds upon the foundation of [browser-use](https://github.com/browser-use/browser-use), which is designed to make websites accessible for AI agents.

We would like to officially thank [WarmShao](https://github.com/warmshao) for his contribution to this project.

**WebUI:** Built on Gradio, supporting most `browser-use` functionalities. This UI is designed to be user-friendly and enables easy interaction with the browser agent.

**Expanded LLM Support:** We've integrated support for various Large Language Models (LLMs), including: Google, OpenAI, Azure OpenAI, Anthropic, DeepSeek, Ollama, etc. We plan to add support for even more models in the future.

**Custom Browser Support:** You can use your own browser with our tool, eliminating the need to re-login to sites or deal with other authentication challenges. This feature also supports high-definition screen recording.

**Persistent Browser Sessions:** You can choose to keep the browser window open between AI tasks, allowing you to see the complete history and state of AI interactions.

### Development Mode of gradio
For development with auto-reload:
```bash
gradio webui.py --watch src
```
This will automatically reload the browser when you make changes to files in the `src` directory.