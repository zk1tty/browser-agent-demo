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

(Recording demo is coming soon....!!!)

Follow [me on Twitter](https://x.com/n0rizkitty) for the latest updates!

---
# Web App

<div style="display: flex; flex-direction: column; align-items: flex-start;">
  <span style="font-size: 1.2em;"><strong>► Run Agent with text prompt</strong></span>
  <ul>
    <li>👍 Robust. </li>
    <li>👎 needs long descriptive prompt. slow.</li>
  </ul>
  <img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-agent-tab.png" alt="Rebrowse Agent Tab" width="100%" style="display: block; margin-bottom: 10px;"/>

  <span style="font-size: 1.2em;"><strong>► Run Agent with recording</strong></span>
  <ul>
    <li>👍 Determinictic. 10x fast. 3x Accurate.</li>
    <li>⚠️ AI-powered fall-back is WIP.</li>
  </ul>
  <img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-replay-tab.png" alt="Rebrowse Replay Tab" width="100%" style="display: block;"/>
</div>

---
## Installation Guide

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

- We recommend using uv as recommended in the [browser-use documentation](https://docs.browser-use.com/quickstart).
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
1. Copy the `.env.example` file to a new file named `.env` in the project root.

2. Open `.env` file, and add API key and chrome paths
    ```bash
    # LLM API Keys
    OPENAI_API_KEY=your_openai_api_key
    ```

    ```bash
    # Chrome Configuration
    # Note: If you are not a macOS user, please identify your Chrome application path:
    # - Windows: `C:\Program Files\Google\Chrome\Application\chrome.exe`
    # - Linux: `/usr/bin/google-chrome`
    CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # replace {username}
    CHROME_USER_DATA="/Users/{username}/Library/Application Support/Google/Chrome"
    CHROME_CDP="http://localhost:9222"
    ```
### Step 4: Close all open Chrome tabs. IMPORTANT!!

<img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/step_quit_chrome.png" alt="Quit Chrome" width="350" style="display: block; margin-left: 0;">

Or, run this command.

```
pkill -9 "Google Chrome"
```


### Step 5: Launch the Web Application
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
  
⚠️ I am temporarily making the test profile on the same 
project file, and name it `/custom_chrome_profile` for debugging purposes.

1. Create a new, clean Chrome profile
    ```
    mkdir -p custom_chrome_profile
    ```

2. Run this Chrome using this new profile.
    Configuration details to connect to your own Chrome are [here](https://docs.browser-use.com/customize/real-browser). 
    
    ```bash
    sh run_custom_chrome.sh
    ```

3. Make sure that the DevTools WebSocket session is available
    ```
    DevTools listening on ws://127.0.0.1:9222/devtools/browser/1bbd94a9-aed9-4462-bc25-fddec9d9663c
    ```
4. Log in to web apps on the new Chrome profile
   In this process, you will create a new Chrome profile to be used by browser-agents.
    - log in to your web accounts: X, LinkedIn, Youtube, etc.
    - If the browser agent can skip this process, it will be easier to handle executions.

### Step 8: Ready to play ^^
- Go to "Choose Agent" tab
- Pick up a Preset Task
- Try "Run Agent"
- Enjoy your browsing agent.

---
## Are you a dev?

Let me navigate you through the technical concept and objectives of Rebrowse.


### Background

This project builds upon the foundation of [browser-use](https://github.com/browser-use/browser-use), which is designed to make websites accessible for AI agents.

The original creator is [WarmShao](https://github.com/warmshao), who made the WebUI.

### Key difference from Browser-use?

I needed to make the workflow of agent behaibier by faster and accurate.   
One approach to do this is recoeding and make it repeatable.   
I introduced our technical architecture in [Architecure](./doc/architecture.md).   

Let me share our technical roadmap.   

- [x] Make a workflow traceable 
- [x] Implement Replay mode with `TraceReplayer` 
- [ ] Add an eval process to trigger Drift or not.  <- enough flexible
- [ ] Take a demo video and add here.
- [ ] Measure the accuracy and speed of replay mode in comparison with agent mode
- [ ] Design `TraceReplayer` memory to handle edge cases
- [ ] Add multi-thread execution of a single workflow
- [ ] Add a reasoning process before the workflow is traceable

### Update on 19th May

I found that @browser-use team released [workflow-use](https://github.com/browser-use/workflow-use).
I'm still researching their approach and objectives.   
Let's talk more on [X](https://x.com/n0rizkitty) or [Telegram](https://x.com/n0rizkitty).

## Product Roadmap

We are going to build a marketplace, where users can share cross-app workflows by recording, instead of node editors like Zapier or n8n.      
Check out our [Roadmap](./doc/ROADMAP.md).   

<img src="https://raw.githubusercontent.com/zk1tty/rebrowse-app/main/assets/rebrowse-flywheel.png" alt="Rebrowse Flywheel" width="350" style="display: block; margin-left: 0;">


### Gradio Development Mode
For development with auto-reload:
```bash
gradio webui.py --watch src
```
This will automatically reload the browser when you make changes to files in the `src` directory.