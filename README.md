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

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-agent-tab.png" alt="Rebrowse Agent Tab" width="100%" style="display: block; margin-bottom: 10px;"/>
  <img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-replay-tab.png" alt="Rebrowse Record Tab" width="100%" style="display: block;"/>
</div>

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

### Step 3: Configure Environment Variables
1. Copy `.env.examlpe` file to `.env` file in the project root.

2. Open `.env` file, and add API key and chrome paths
    ```bash
    # LLM API Keys
    OPENAI_API_KEY=your_openai_api_key
    ```

    ```bash
    # Chrome Configuration
    # Note: Not MacOS user? please identify your Chrome application path:
    # - Windows: `C:\Program Files\Google\Chrome\Application\chrome.exe`
    # - Linux: `/usr/bin/google-chrome`
    CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # replace {username}
    CHROME_USER_DATA="/Users/{username}/Library/Application Support/Google/Chrome"
    CHROME_CDP="http://localhost:9222"
    ```
### Step 4: Close all opening Chrome tabs!! IMPORTANT!!

<img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/step_quit_chrome.png" alt="Quit Chrome" width="350" style="display: block; margin-left: 0;">

Or, run this command.

```
pkill -9 "Google Chrome"
```


### Step 5: Launch the web Application
```bash
# Start with Gradio
gradio webui.py

# Or run directly with Python
python webui.py
```
The application will be available at `http://127.0.0.1:7860`

### Step 6: Use FireFox or Safari to open 127.0.0.1:7860
- DO NOT use Chrome to open this web app.
- Open Safari or FIrefox, and go to "http://127.0.0.1:7860"


### Step 7: Run a clean Chrome with CDP session
  
⚠️ I'm temporary making the test profile on the same 
project file, and name it `/custom_chrome_profile` for debuging purpose.

1. create a new clean chrome profile
    ```
    mkdir -p custom_chrome_profile
    ```

2. Run this Chrome using this new profile.
    Configuration details to conenct to your own Chrome is [here](https://docs.browser-use.com/customize/real-browser). 
    
    ```bash
    sh run_custom_chrome.sh
    ```

3. Make sure that devtool websocket session is available
    ```
    DevTools listening on ws://127.0.0.1:9222/devtools/browser/1bbd94a9-aed9-4462-bc25-fddec9d9663c
    ```
4. Login web apps on a new Chrome profile
   In this process, you will create a new Chrome profile to be used by browser-agents.
    - log in to your web accounts: X, LinkedIn, Youtube, etc.
    - if browser-agent can skip this process, then easier to handle executions.

### Step 8: Ready to play ^^
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