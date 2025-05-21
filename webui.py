from src.browser.custom_browser import CustomBrowser
import pdb
import logging
import os

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse

# Import task templates
from task_templates import TASK_TEMPLATES

# TODO: add logging configure
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# For more detailed agent logging
logging.getLogger('src.agent.custom_agent').setLevel(logging.INFO)
logging.getLogger('src.utils.replayer').setLevel(logging.DEBUG)
logging.getLogger('browser_use.agent.service').setLevel(logging.INFO) # For the base agent
logging.getLogger('browser_use.controller.service').setLevel(logging.INFO)
logging.getLogger('browser_use.browser').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

import gradio as gr
import inspect
from functools import wraps
from gradio.components import Component

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot, MissingAPIKeyError
from src.utils import utils
from src.browser.custom_context_config import CustomBrowserContextConfig as AppCustomBrowserContextConfig
from browser_use.browser.context import BrowserContextConfig
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any

# Global variables for persistence
_global_browser: Optional[CustomBrowser] = None
_global_browser_context: Optional[CustomBrowserContext] = None
_global_agent = None
_global_input_tracking_active = False

# Create the global agent state instance
_global_agent_state = AgentState()

# webui config
webui_config_manager = utils.ConfigManager()

# New: For manual record/replay
_last_manual_trace_path: Optional[str] = None
MANUAL_TRACES_DIR = "./tmp/input_tracking"  # Reverted to ./tmp/input_tracking

# New: repeat feature
import json

# New: user input tracking functions
from src.utils import user_input_functions

def context_is_closed(ctx) -> bool:
    """
    Heuristic: accessing ctx.pages on a disposed context raises an exception.
    Works for both sync & async BrowserContext objects.
    """
    try:
        _ = ctx.pages  # attribute exists on both sync/async contexts
        return False
    except Exception:
        return True

def _extract_initial_actions(history_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn the *first* model-generated step in the uploaded browsing history into
    an `initial_actions` list that our Agent understands.
    """
    first = history_json['history'][0]
    return first['model_output']['action']

def scan_and_register_components(blocks):
    """Scan a Blocks object and register all interactive components, excluding buttons"""
    global webui_config_manager

    def traverse_blocks(block, prefix=""):
        registered = 0

        # Process components of the Blocks object
        if hasattr(block, "children"):
            for i, child in enumerate(block.children):
                if isinstance(child, Component):
                    # Exclude Button components
                    if getattr(child, "interactive", False) and not isinstance(child, gr.Button):
                        name = f"{prefix}component_{i}"
                        if hasattr(child, "label") and child.label:
                            # Use label as part of the name
                            label = child.label
                            name = f"{prefix}{label}"
                        logger.debug(f"Registering component: {name}")
                        webui_config_manager.register_component(name, child)
                        registered += 1
                elif hasattr(child, "children"):
                    # Recursively process nested Blocks
                    new_prefix = f"{prefix}block_{i}_"
                    registered += traverse_blocks(child, new_prefix)

        return registered

    total = traverse_blocks(blocks)
    logger.info(f"Total registered components: {total}")

def save_current_config():
    return webui_config_manager.save_current_config()


def update_ui_from_config(config_file):
    return webui_config_manager.update_ui_from_config(config_file)


def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text

    import re

    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)

    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)

    return result


async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent

    try:
        if _global_agent is not None:
            logger.debug("WebUI: Calling _global_agent.stop()") # DEBUG
            _global_agent.stop()
            if hasattr(_global_agent, 'state') and _global_agent.state is not None:
                 logger.debug(f"WebUI: Agent state after stop() call: stopped={_global_agent.state.stopped}, paused={_global_agent.state.paused}") # DEBUG
            else:
                 logger.debug("WebUI: _global_agent or _global_agent.state is None after stop call attempt.")

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def stop_research_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (  # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens,
        enable_input_tracking=False,
        save_input_tracking_path=MANUAL_TRACES_DIR
):
    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens,
                enable_input_tracking=enable_input_tracking,
                save_input_tracking_path=save_input_tracking_path
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens,
                enable_input_tracking=enable_input_tracking,
                save_input_tracking_path=save_input_tracking_path
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        # latest_video = None
        # if save_recording_path:
        #     new_videos = set(
        #         glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
        #         + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
        #     )
        #     if new_videos - existing_videos:
        #         latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )

    except MissingAPIKeyError as e:
        logger.error(str(e))
        raise gr.Error(str(e), print_exception=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',  # final_result
            errors,  # errors
            '',  # model_actions
            '',  # model_thoughts
            None,  # latest_video
            None,  # history_file
            None,  # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )

async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens,
        enable_input_tracking: bool = False,
        save_input_tracking_path: str = MANUAL_TRACES_DIR
):
    try:
        global _global_browser, _global_browser_context, _global_agent
        
        # Update the browser context reference in user_input_functions
        user_input_functions.set_browser_context(_global_browser_context)

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [
                    f"--user-data-dir={chrome_user_data}",
                    "--profile-directory=Default",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-web-security",
                    "--disable-site-isolation-trials"
                ]
        else:
            chrome_path = None

        if _global_browser is None:
            logger.info("Global browser (CustomBrowser) not found for org agent, initializing...")
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )
            await _global_browser.async_init()
        elif not (_global_browser.playwright_browser and _global_browser.playwright_browser.is_connected()):
            logger.info("Global CustomBrowser found but not connected for org agent. Re-initializing...")
            await _global_browser.async_init()

        if _global_browser_context is None:
            logger.info("Global browser context not found for org agent run, initializing...")
            # _global_browser is now guaranteed to be CustomBrowser (or init failed before this)
            context_config = AppCustomBrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path if save_recording_path else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(
                    width=window_w, height=window_h
                ),
                enable_input_tracking=enable_input_tracking, 
                save_input_tracking_path=save_input_tracking_path
            )
            _global_browser_context = await _global_browser.new_context(config=context_config)
        
        # If agent's own input tracking is enabled and no pages exist, open one.
        if (enable_input_tracking and 
            isinstance(_global_browser_context, CustomBrowserContext) and 
            _global_browser_context.playwright_context is not None and # Explicitly check playwright_context for None
            not _global_browser_context.playwright_context.pages):
            logger.info("Agent run has input tracking enabled and no pages exist. Opening a new default page.")
            await _global_browser_context.playwright_context.new_page()

        if _global_agent is None:
            logger.info(f"OrgAgent Task: {task}")
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        # Handle case where history might be None (e.g., if it were a replay, though not via this path)
        if history is None:
            logger.warning("Agent run in run_org_agent returned None for history. Returning Nones.")
            return None, None, None, None, None, None

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens,
        enable_input_tracking,
        save_input_tracking_path
):
    try:
        global _global_browser, _global_browser_context, _global_agent
        
        # Configure browser settings
        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp
        chrome_path = None
        chrome_user_data = None

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                # Add Chrome-specific flags (excluding user-data-dir as it's handled by launch_persistent_context)
                extra_chromium_args += [
                    "--profile-directory=Default",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-web-security",
                    "--disable-site-isolation-trials"
                ]

        # Initialize global browser if needed
        if _global_browser is None:
            logger.info("Global browser (CustomBrowser) not found, initializing for custom agent...")
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )
            await _global_browser.async_init() 
        elif not (_global_browser.playwright_browser and _global_browser.playwright_browser.is_connected()):
            logger.info("Global CustomBrowser found but not connected for custom agent. Re-initializing...")
            await _global_browser.async_init()
        # No need to check if it's base Browser, as it's now always initialized as CustomBrowser if None

        if _global_browser_context is None:
            logger.info("Global browser context not found for custom agent run, initializing...")
            if use_own_browser and _global_browser and _global_browser.config and _global_browser.config.cdp_url:
                try:
                    logger.info(f"Attempting to reuse existing browser context via CDP: {_global_browser.config.cdp_url}")
                    _global_browser_context = await _global_browser.reuse_existing_context()
                    logger.info(f"Successfully reused existing browser context: {_global_browser_context}")
                    if _global_browser_context and _global_browser_context.playwright_context and not _global_browser_context.playwright_context.pages:
                        logger.warning("Reused context has no pages. Opening a new blank tab in it for recording.")
                        page = await _global_browser_context.playwright_context.new_page()
                        await page.goto("about:blank") # Or a user-configurable start page
                        await page.bring_to_front()
                except Exception as e:
                    logger.error(f"Failed to reuse existing browser context: {e}. Falling back to new context strategy if possible.")
                    # Ensure _global_browser_context is None so fallback can occur if this path was the primary attempt
                    _global_browser_context = None 
                    # Re-raise or handle more gracefully depending on desired UX
                    # For now, if reuse fails, it will fall through to new context creation or fail if browser not init
                    pass # Allow to fall through to new context creation if reuse fails badly
            
            # If not using CDP or reuse failed and _global_browser_context is still None
            if _global_browser_context is None:
                logger.info("Initializing new browser context as not using CDP, or reuse failed.")
                # Ensure _global_browser is initialized if it wasn't for some reason (should be by prior logic)
                if not (_global_browser and _global_browser.playwright): # Check if browser is alive
                     logger.error("Global browser not available for creating new context. Cannot proceed with recording setup.")
                     return "Status: Error - Browser not available", gr.update(interactive=True), gr.update(interactive=False), None

                _global_browser_context = await _global_browser.new_context(
                    config=AppCustomBrowserContextConfig(
                        enable_input_tracking=True, 
                        save_input_tracking_path=MANUAL_TRACES_DIR, # This comes from global
                        browser_window_size=BrowserContextWindowSize(width=1280, height=1100) # TODO: Get from UI?
                    )
                )
                if _global_browser_context and _global_browser_context.playwright_context:
                    logger.info("New browser context created for recording. Opening a new page in it.")
                    try:
                        page = await _global_browser_context.playwright_context.new_page()
                        await page.bring_to_front()
                        await page.goto("https://www.google.com") # Default for new context
                        logger.warning(f"Record Tab: A new browser context and page ('{page.url}') have been created, navigated to Google, and focused. Please use THIS page for recording.")
                    except Exception as e:
                        logger.error(f"Error opening, navigating, or focusing new page for new context recording: {e}")
                else:
                    logger.error("Failed to create a new browser context properly for recording.")
                    return "Status: Error - No valid browser context (check logs)", gr.update(interactive=True), gr.update(interactive=False), None
        
        # If agent's own input tracking is enabled and no pages exist, open one.
        if (enable_input_tracking and 
            isinstance(_global_browser_context, CustomBrowserContext) and 
            _global_browser_context.playwright_context is not None and # Explicitly check playwright_context for None
            not _global_browser_context.playwright_context.pages):
            logger.info("Agent run has input tracking enabled and no pages exist. Opening a new default page.")
            await _global_browser_context.playwright_context.new_page()

        # Set the context for user_input_functions
        user_input_functions.set_browser_context(_global_browser_context)

        controller = CustomController()

        # Create and run agent
        if _global_agent is None:
            logger.info(f"CustomAgent Task: {task}")
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(task_input=task, max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        # Handle case where history might be None (e.g., if CustomAgent was run in replay mode directly)
        if history is None:
            logger.warning("CustomAgent run returned None for history. Returning Nones.")
            # This path is unlikely if run_custom_agent is always called with a string task for autonomous mode.
            return None, None, None, None, None, None

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

# main Custom Agent on click
async def run_with_stream(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens,
        enable_input_tracking=False,
        save_input_tracking_path=MANUAL_TRACES_DIR
):
    global _global_agent

    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            chrome_cdp=chrome_cdp,
            max_input_tokens=max_input_tokens,
            enable_input_tracking=enable_input_tracking,
            save_input_tracking_path=save_input_tracking_path
        )
        # Add HTML content at the start of the result array
        yield [gr.update(visible=False)] + list(result)
    else:
        try:
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_num_ctx=llm_num_ctx,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method,
                    chrome_cdp=chrome_cdp,
                    max_input_tokens=max_input_tokens,
                    enable_input_tracking=enable_input_tracking,
                    save_input_tracking_path=save_input_tracking_path
                )
            )

            # Initialize values for streaming
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            recording_gif = trace = history_file = None

            # Periodically update the stream while the agent task is running
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception as e:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent and _global_agent.state.stopped:
                    yield [
                        gr.HTML(value=html_content, visible=True),
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        recording_gif,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),  # stop_button
                        gr.update(interactive=False),  # run_button
                    ]
                    break
                else:
                    yield [
                        gr.HTML(value=html_content, visible=True),
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        recording_gif,
                        trace,
                        history_file,
                        gr.update(),  # Re-enable stop button
                        gr.update()  # Re-enable run button
                    ]
                await asyncio.sleep(0.1)

            # Once the agent task completes, get the results
            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, recording_gif, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                recording_gif = trace = history_file = None

            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                gr.HTML(value=html_content, visible=True),
                final_result,
                errors,
                model_actions,
                model_thoughts,
                recording_gif,
                trace,
                history_file,
                stop_button,
                run_button
            ]

        except Exception as e:
            import traceback
            yield [
                gr.HTML(
                    value=f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                    visible=True),
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True)  # Re-enable run button
            ]

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}


async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None


async def run_deep_search(research_task, max_search_iteration_input, max_query_per_iter_input, llm_provider,
                          llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision,
                          use_own_browser, headless, chrome_cdp):
    from src.utils.deep_research import deep_research
    global _global_agent_state

    # Clear any previous stop request
    _global_agent_state.clear_stop()

    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        num_ctx=llm_num_ctx,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    markdown_content, file_path = await deep_research(research_task, llm, _global_agent_state,
                                                      max_search_iterations=max_search_iteration_input,
                                                      max_query_num=max_query_per_iter_input,
                                                      use_vision=use_vision,
                                                      headless=headless,
                                                      use_own_browser=use_own_browser,
                                                      chrome_cdp=chrome_cdp
                                                      )

    return markdown_content, file_path, gr.update(value="Stop", interactive=True), gr.update(interactive=True)


# ✨ NEW – coroutine to replay uploaded browsing history
async def run_repeat(
    history_file,
    max_steps,
    llm_provider, llm_model_name, ollama_num_ctx,
    llm_temperature, llm_base_url, llm_api_key,
    use_own_browser, keep_browser_open, headless, disable_security,
    window_w, window_h, save_recording_path, save_agent_history_path,
    save_trace_path, enable_recording, chrome_cdp, use_vision,
    max_actions_per_step, tool_calling_method, max_input_tokens
) -> Tuple[str, str]:
    """
    1. Parse the JSON.
    2. Build `initial_actions`.
    3. Fire up the regular custom Agent, passing `initial_actions`.
    """
    try:
        # If already a dict (parsed JSON from Gradio)
        if isinstance(history_file, dict):
            history_json = history_file
        # If NamedString or UploadFile with .name path
        elif hasattr(history_file, 'name') and isinstance(history_file.name, str) and os.path.exists(history_file.name):
            with open(history_file.name, 'r') as f:
                history_json = json.load(f)
        # If file-like object supports read()
        elif hasattr(history_file, 'read') and callable(history_file.read):
            history_json = json.load(history_file)
        # If it's a string: try raw JSON, else file path
        elif isinstance(history_file, str):
            try:
                history_json = json.loads(history_file)
            except json.JSONDecodeError:
                with open(history_file, 'r') as f:
                    history_json = json.load(f)
        else:
            # Fallback: parse as text
            history_json = json.loads(str(history_file))
        initial_actions = _extract_initial_actions(history_json)
    except Exception as e:
        return "", f"❌ Failed to parse JSON: {e}"

    # Re-use run_browser_agent under the hood
    result, errors, *_ = await run_browser_agent(
        agent_type="custom",
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_num_ctx=ollama_num_ctx,
        llm_temperature=llm_temperature,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        use_own_browser=use_own_browser,
        keep_browser_open=keep_browser_open,
        headless=headless,
        disable_security=disable_security,
        window_w=window_w,
        window_h=window_h,
        save_recording_path=save_recording_path,
        save_agent_history_path=save_agent_history_path,
        save_trace_path=save_trace_path,
        enable_recording=enable_recording,
        task="Repeat uploaded browsing session",
        add_infos="Running in replay mode.",
        max_steps=int(max_steps),
        use_vision=use_vision,
        max_actions_per_step=max_actions_per_step,
        tool_calling_method=tool_calling_method,
        chrome_cdp=chrome_cdp,
        max_input_tokens=max_input_tokens
    )
    return result, errors

# New: coroutine to start input tracking with context
async def start_input_tracking_with_context():
    global _global_browser, _global_browser_context, _global_input_tracking_active, _last_manual_trace_path

    status_update = ""
    trace_path_update = "" # Trace path is known after stopping, not starting
    error_message = ""
    start_button_interactive = True
    stop_button_interactive = False

    try:
        logger.info("Attempting to start input tracking...")

        async with async_playwright() as p: # p is not directly used below with CustomBrowser based on other funcs
            browser_needs_init = _global_browser is None
            if not browser_needs_init:
                # Corrected attribute: _playwright_browser -> playwright_browser
                if _global_browser and (_global_browser.playwright_browser is None or not _global_browser.playwright_browser.is_connected()):
                    logger.info("Global browser's Playwright browser instance is not connected or missing. Re-initializing browser.")
                    browser_needs_init = True
            
            if browser_needs_init:
                logger.info("Global browser not initialized or needs re-initialization. Initializing...")
                # Use CHROME_CDP_URL for the full URL, CHROME_REMOTE_DEBUG_PORT is for the script
                cdp_full_url = os.getenv("CHROME_CDP_URL")
                logger.info(f"Retrieved CHROME_CDP_URL for recording: '{cdp_full_url}'") # ADDED LOGGING
                if cdp_full_url:
                    logger.info(f"Attempting to connect to existing browser via CDP URL: {cdp_full_url}")
                
                browser_config = BrowserConfig(
                    headless=False, 
                    cdp_url=cdp_full_url # Use the full CDP URL from env
                )
                _global_browser = CustomBrowser(config=browser_config)
                await _global_browser.async_init()
                
                # Ensure this check and status update are present
                if not _global_browser or not _global_browser.playwright: 
                    logger.error("Browser instantiation/initialization failed.")
                    error_message = "Browser initialization failed."
                    return gr.update(value=error_message), gr.update(value=_last_manual_trace_path or "No trace yet"), gr.update(interactive=True), gr.update(interactive=False)
                status_update += "Browser initialized. "
                logger.info("Browser initialized.")
            else:
                logger.info("Using existing global browser.")

            if not (_global_browser and _global_browser.playwright):
                logger.error("Playwright instance within CustomBrowser is not available after init/create.")
                error_message = "Browser initialization failed (Playwright linkage). Cannot start tracking."
                return gr.update(value=error_message), gr.update(value=_last_manual_trace_path or "No trace yet"), gr.update(interactive=True), gr.update(interactive=False)

            context_needs_init = _global_browser_context is None
            if not context_needs_init:
                # Add explicit check for _global_browser_context before accessing its attributes
                if _global_browser_context and (_global_browser_context.playwright_context is None or \
                   context_is_closed(_global_browser_context.playwright_context)):
                    logger.info("Global browser context is unusable (closed or no Playwright context). Re-initializing context.")
                    _global_browser_context = None # Force re-init by nullifying
                    context_needs_init = True

            if context_needs_init: # This means _global_browser_context is None or marked for re-init
                logger.info("Global browser context needs initialization.")
                
                # Check if the _global_browser was connected via CDP
                attempt_reuse = False
                if _global_browser and _global_browser.config and _global_browser.config.cdp_url:
                    logger.info(f"Browser was connected via CDP ({_global_browser.config.cdp_url}). Attempting to reuse existing context.")
                    attempt_reuse = True
                
                if attempt_reuse:
                    _global_browser_context = await _global_browser.reuse_existing_context()
                    if _global_browser_context:
                        logger.info(f"Successfully reused existing context: {_global_browser_context}")
                        # Ensure the reused context has pages
                        if not _global_browser_context.pages: # pages property calls _ctx.pages
                             logger.warning("Reused context has no pages. Creating one.")
                             try:
                                 await _global_browser_context.new_page() # Calls _ctx.new_page()
                             except Exception as e:
                                 logger.error(f"Error creating page in reused context: {e}")
                                 _global_browser_context = None # Mark reuse as failed
                    else:
                        logger.warning("Failed to reuse existing context or no suitable context found.")
                
                # If reuse was not attempted, or failed (so _global_browser_context is still None)
                if not _global_browser_context: 
                    if attempt_reuse: # Log if reuse was tried and failed
                        logger.info("Falling back to creating a new browser context after failed reuse attempt.")
                    else: # Log if reuse was not attempted (e.g. no CDP)
                        logger.info("Proceeding to create a new browser context.")
                    
                    context_config_object = AppCustomBrowserContextConfig(
                        # Defaults from AppCustomBrowserContextConfig will be used.
                        # Specific settings like 'enable_input_tracking' are handled by explicit calls later.
                    )
                    if not _global_browser: 
                         logger.error("Cannot create new context, _global_browser is None. This should not happen here.")
                         error_message = "Critical error: Browser object became None before context creation."
                         return gr.update(value=error_message), gr.update(value=_last_manual_trace_path or "No trace yet"), gr.update(interactive=True), gr.update(interactive=False)
                    
                    _global_browser_context = await _global_browser.new_context(config=context_config_object)
                    
                    if _global_browser_context:
                        logger.info("New browser context created.")
                        # Ensure new context has a page (as per original logic)
                        if not _global_browser_context.pages:
                            logger.info("Newly created context has no pages. Creating one.")
                            try:
                                await _global_browser_context.new_page()
                            except Exception as e:
                                logger.error(f"Error creating page in new context: {e}")
                                # This could be a critical failure for the new context
                    else:
                        logger.error("Failed to create a new browser context.")
                        # Error handling for failed new context creation needed here
                        error_message = "Failed to create new browser context."
                        return gr.update(value=error_message), gr.update(value=_last_manual_trace_path or "No trace yet"), gr.update(interactive=True), gr.update(interactive=False)


                # Check if context creation/reuse was successful and the context is valid
                if not _global_browser_context: # Covers failure of both reuse and new creation paths
                    logger.error("Browser context is None after initialization attempts.")
                    error_message = error_message or "Browser context could not be established."
                    return gr.update(value=error_message), gr.update(value=_last_manual_trace_path or "No trace yet"), gr.update(interactive=True), gr.update(interactive=False)
                
                # Now that _global_browser_context is confirmed to be not None, check its playwright_context
                if not _global_browser_context.playwright_context:
                    logger.error("Browser context was established, but its internal Playwright context is missing.")
                    error_message = "Browser context is invalid (missing Playwright link)."
                    return gr.update(value=error_message), gr.update(value=_last_manual_trace_path or "No trace yet"), gr.update(interactive=True), gr.update(interactive=False)

                # Original check (now split into the two above for clarity and safety):
                # if not (_global_browser_context and _global_browser_context.playwright_context):
                #     logger.error("Failed to create or initialize browser context.")

                current_pages = _global_browser_context.pages 
                if not current_pages:
                    logger.info("New context has no pages. Creating one.")
                    await _global_browser_context.new_page()
                    logger.info("New page created in new context.")
                status_update += "Browser context initialized. "
                logger.info("Browser context initialized.")
            else:
                logger.info("Using existing global browser context.")

        if not _global_browser_context:
            error_message = "Browser context is not available after setup."
            logger.error(error_message)
        elif _global_input_tracking_active:
            status_update = "Input tracking is already active."
            logger.warning(status_update)
            start_button_interactive = False
            stop_button_interactive = True
        else:
            if not os.path.exists(MANUAL_TRACES_DIR):
                os.makedirs(MANUAL_TRACES_DIR, exist_ok=True)
            
            logger.info(f"Setting browser context for user_input_functions: {_global_browser_context}")
            user_input_functions.set_browser_context(_global_browser_context)
            
            logger.info("Calling user_input_functions.start_input_tracking()...")
            func_status_msg, _, _ = await user_input_functions.start_input_tracking()
            logger.info(f"user_input_functions.start_input_tracking() returned: {func_status_msg}")

            if "error" not in func_status_msg.lower() and "failed" not in func_status_msg.lower():
                _global_input_tracking_active = True
                status_update += f" Input tracking started. Status: {func_status_msg}"
                trace_path_update = "Recording... (Path will be shown on stop)"
                logger.info(status_update)
                start_button_interactive = False
                stop_button_interactive = True
            else:
                error_message = f"Failed to start tracking: {func_status_msg}"
                logger.error(error_message)

    except MissingAPIKeyError as e:
        logger.error(f"Missing API Key: {e}")
        error_message = f"Configuration Error: {e}. Please check your environment variables or config files."
    except Exception as e:
        logger.error(f"Error in start_input_tracking_with_context: {e}", exc_info=True)
        error_message = f"An unexpected error occurred: {str(e)}"
    
    if error_message and not status_update.endswith(error_message) and not status_update.startswith(error_message):
        final_status = f"{status_update} Error: {error_message}" if status_update else error_message
    elif error_message:
        final_status = error_message
    else:
        final_status = status_update

    if error_message:
        start_button_interactive = True 
        stop_button_interactive = False

    return (
        gr.update(value=final_status), # For input_track_status
        gr.update(interactive=start_button_interactive), # For input_track_start_btn
        gr.update(interactive=stop_button_interactive),  # For input_track_stop_btn
        gr.update(value=trace_path_update or _last_manual_trace_path or "No trace yet") # For trace_file_path
    )

async def stop_input_tracking_with_context():
    global _global_browser_context, _global_input_tracking_active, _last_manual_trace_path

    if not _global_browser_context or not _global_input_tracking_active:
        logger.warning("Input tracking not active or browser context not available.")
        # This path needs to return 5 values to match Gradio's expectation
        return (
            "Tracking not active or context unavailable.", 
            gr.update(interactive=True), 
            gr.update(interactive=False), 
            _last_manual_trace_path,
            {"message": "Tracking not active or context unavailable."} # Added missing 5th value for recorded_trace_info_display
        )
    
    try:
        logger.info("Attempting to stop user input tracking via CustomBrowserContext...")
        filepath = await _global_browser_context.stop_input_tracking()
        _global_input_tracking_active = False 
        _last_manual_trace_path = filepath 
        status_message = f"Input tracking stopped. Trace saved to: {filepath}" if filepath else "Input tracking stopped. No file saved."
        logger.info(status_message)
        
        # Get trace info for the JSON display
        trace_info = {}
        if filepath:
            try:
                trace_info = user_input_functions.get_file_info(filepath)
            except Exception as e_info:
                logger.error(f"Error getting trace info for display: {e_info}")
                trace_info = {"error": f"Could not load trace info: {str(e_info)}"}
        else:
            trace_info = {"message": "No trace file was saved."}

        return (
            status_message, 
            gr.update(value="▶️ Start Recording", interactive=True), 
            gr.update(value="⏹️ Stop Recording", interactive=False), 
            filepath,
            trace_info # This is the 5th value for recorded_trace_info_display
        )
    except Exception as e:
        logger.error(f"Exception during stop_input_tracking_with_context: {e}", exc_info=True)
        # This path also needs to return 5 values
        return (
            f"Error stopping input tracking: {e}", 
            gr.update(interactive=True), 
            gr.update(interactive=True), 
            _last_manual_trace_path,
            {"error": f"Error stopping tracking: {str(e)}"} # Added missing 5th value
        )


def create_ui(theme_name="Citrus"):
    css = """
    .gradio-container {
        width: 60vw !important; 
        max-width: 60% !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    /* Add styles for resizable textbox */
    .resizable-textbox {
        resize: both !important;
        min-height: 100px !important;
        overflow: auto !important;
    }
    /* Styles for the trace table */
    .trace-table {
        width: 100%;
        border-collapse: collapse;
    }
    .trace-table th, .trace-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .trace-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .trace-table th {
        padding-top: 12px;
        padding-bottom: 12px;
        background-color: #4CAF50;
        color: white;
    }
    """

    with gr.Blocks(
            title="Rebrowse", theme=theme_map[theme_name], css=css
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # Rebrowse
                ### the browser-agent which repeats cross-app workflows by recording
                """,
                elem_classes=["header-text"],
            )
        
        # Define trace_file_path here to be in scope for all tabs that might output to it
        trace_file_path = gr.Textbox( 
            label="Selected Trace File Path (for Replay)", 
            value="",
            interactive=False,
            visible=False # Set to False to hide it from global rendering
        )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Settings", id=2):
                with gr.Group():
                    gr.Markdown("### Agent Settings")
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value="custom",
                        info="Select the type of agent to use",
                        interactive=True
                    )
                    with gr.Column():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=100,
                            step=1,
                            label="Max Run Steps",
                            info="Maximum number of steps the agent will take",
                            interactive=True
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1,
                            label="Max Actions per Step",
                            info="Maximum number of actions the agent will take per step",
                            interactive=True
                        )
                    with gr.Column():
                        use_vision = gr.Checkbox(
                            label="Use Vision",
                            value=True,
                            info="Enable visual processing capabilities",
                            interactive=True
                        )
                        max_input_tokens = gr.Number(
                            label="Max Input Tokens",
                            value=128000,
                            precision=0,
                            interactive=True
                        )
                        tool_calling_method = gr.Dropdown(
                            label="Tool Calling Method",
                            value="auto",
                            interactive=True,
                            allow_custom_value=True,
                            choices=["auto", "json_schema", "function_calling"],
                            info="Tool Calls Function Name",
                            visible=False
                        )

                with gr.Group():
                    gr.Markdown("### LLM Settings")
                    llm_provider = gr.Dropdown(
                        choices=[provider for provider, model in utils.model_names.items()],
                        label="LLM Provider",
                        value="openai",
                        info="Select your preferred language model provider",
                        interactive=True
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        choices=utils.model_names['openai'],
                        value="gpt-4o",
                        interactive=True,
                        allow_custom_value=True,
                        info="Select a model in the dropdown options or directly type a custom model name"
                    )
                    ollama_num_ctx = gr.Slider(
                        minimum=2 ** 8,
                        maximum=2 ** 16,
                        value=16000,
                        step=1,
                        label="Ollama Context Length",
                        info="Controls max context length model needs to handle (less = faster)",
                        visible=False,
                        interactive=True
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.6,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs",
                        interactive=True
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value="",
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value="",
                            info="Your API key (leave blank to use .env)"
                        )

                with gr.Group():
                    gr.Markdown("### Browser Settings")
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=True,
                            info="Use your existing browser instance",
                            interactive=True
                        )
                        keep_browser_open = gr.Checkbox(
                            label="Keep Browser Open",
                            value=True,
                            info="Keep Browser Open between Tasks",
                            interactive=True
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=False,
                            info="Run browser without GUI",
                            interactive=True
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=True,
                            info="Disable browser security features",
                            interactive=True
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=True,
                            info="Enable saving browser recordings",
                            interactive=True
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=1280,
                            info="Browser window width",
                            interactive=True
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=1100,
                            info="Browser window height",
                            interactive=True
                        )

                    chrome_cdp = gr.Textbox(
                        label="CDP URL",
                        placeholder="http://localhost:9222",
                        value="",
                        info="CDP for google remote debugging",
                        interactive=True,
                    )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value="./tmp/record_videos",
                        info="Path to save browser recordings",
                        interactive=True,
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value="./tmp/traces",
                        info="Path to save Agent traces",
                        interactive=True,
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value="./tmp/agent_history",
                        info="Specify the directory where agent history should be saved.",
                        interactive=True,
                    )
                    
                    # User input tracking settings
                    enable_input_tracking = gr.Checkbox(
                        label="Enable User Input Tracking",
                        value=False,
                        info="Enable tracking of user mouse and keyboard inputs for recording workflows",
                        interactive=True
                    )
                    
                    save_input_tracking_path = gr.Textbox(
                        label="Input Tracking Save Path",
                        placeholder="e.g., ./tmp/input_tracking",
                        value=MANUAL_TRACES_DIR,
                        info="Specify the directory where user input tracking files should be saved.",
                        interactive=True,
                    )

            with gr.TabItem("🤖 Prompt Agent", id=1):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Preset Tasks")
                        with gr.Row():
                            post_to_x = gr.Button("➡️ Post Grok-gen AI news on X", variant="secondary")
                            heygen_video_button = gr.Button("🎥 Generate Video with HeyGen", variant="secondary")
                            custom_task = gr.Button("Custom Task", variant="secondary")

                task = gr.Textbox(
                    label="Task Description",
                    lines=8,
                    placeholder="Enter your task here...",
                    value="",
                    info="Describe what you want the agent to do",
                    interactive=True,
                    elem_classes=["resizable-textbox"]
                )

                # Define preset task templates
                def set_post_to_x():
                    return TASK_TEMPLATES["post_to_x"]

                def set_heygen_video():
                    return TASK_TEMPLATES["heygen_video"]

                def set_custom_task():
                    return TASK_TEMPLATES["custom_task"]

                # Bind the preset task buttons
                post_to_x.click(fn=set_post_to_x, outputs=task)
                custom_task.click(fn=set_custom_task, outputs=task)

                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                    value="",
                    interactive=True
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

                with gr.Row():
                    browser_view = gr.HTML(
                        value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
                        label="Live Browser View",
                        visible=False
                    )

                gr.Markdown("### Results")
                with gr.Row():
                    with gr.Column():
                        final_result_output = gr.Textbox(
                            label="Final Result", lines=3, show_label=True
                        )
                    with gr.Column():
                        errors_output = gr.Textbox(
                            label="Errors", lines=3, show_label=True
                        )
                with gr.Row():
                    with gr.Column():
                        model_actions_output = gr.Textbox(
                            label="Model Actions", lines=3, show_label=True, visible=False
                        )
                    with gr.Column():
                        model_thoughts_output = gr.Textbox(
                            label="Model Thoughts", lines=3, show_label=True, visible=False
                        )
                recording_gif = gr.Image(label="Result GIF", format="gif")
                trace_file = gr.File(label="Trace File")
                agent_history_file = gr.File(label="Agent History")

            with gr.TabItem("🧐 Deep Research", id=3, visible=False):
                research_task_input = gr.Textbox(label="Research Task", lines=5,
                                                 value="Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects, substantiated with examples of relevant models and techniques. The report should reflect original insights and analysis, moving beyond mere summarization of existing literature.",
                                                 interactive=True)
                with gr.Row():
                    max_search_iteration_input = gr.Number(label="Max Search Iteration", value=3,
                                                           precision=0,
                                                           interactive=True)  # precision=0 确保是整数
                    max_query_per_iter_input = gr.Number(label="Max Query per Iteration", value=1,
                                                         precision=0,
                                                         interactive=True)  # precision=0 确保是整数
                with gr.Row():
                    research_button = gr.Button("▶️ Run Deep Research", variant="primary", scale=2)
                    stop_research_button = gr.Button("⏹ Stop", variant="stop", scale=1)
                markdown_output_display = gr.Markdown(label="Research Report")
                markdown_download = gr.File(label="Download Research Report")

            # Bind the stop button click event after errors_output is defined
            stop_button.click(
                fn=stop_agent,
                inputs=[],
                outputs=[stop_button, run_button],
            )

            # Run button click handler
            run_button.click(
                fn=run_with_stream,
                inputs=[
                    agent_type, llm_provider, llm_model_name, ollama_num_ctx, llm_temperature, llm_base_url,
                    llm_api_key,
                    use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                    save_recording_path, save_agent_history_path, save_trace_path,
                    enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step,
                    tool_calling_method, chrome_cdp, max_input_tokens, enable_input_tracking, save_input_tracking_path
                ],
                outputs=[
                    browser_view,  # Browser view
                    final_result_output,  # Final result
                    errors_output,  # Errors
                    model_actions_output,  # Model actions
                    model_thoughts_output,  # Model thoughts
                    recording_gif,  # Latest recording
                    trace_file,  # Trace file
                    agent_history_file,  # Agent history file
                    stop_button,  # Stop button
                    run_button  # Run button
                ],
            )
            
            # Add input tracking checkbox to sync with save path
            enable_input_tracking.change(
                lambda enabled: gr.update(interactive=enabled),
                inputs=enable_input_tracking,
                outputs=save_input_tracking_path
            )

            # Run Deep Research
            research_button.click(
                fn=run_deep_search,
                inputs=[research_task_input, max_search_iteration_input, max_query_per_iter_input, llm_provider,
                        llm_model_name, ollama_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision,
                        use_own_browser, headless, chrome_cdp],
                outputs=[markdown_output_display, markdown_download, stop_research_button, research_button]
            )
            # Bind the stop button click event after errors_output is defined
            stop_research_button.click(
                fn=stop_research_agent,
                inputs=[],
                outputs=[stop_research_button, research_button],
            )

            # temporarly invisible
            with gr.TabItem("🎥 Recordings", id=7, visible=False):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []

                    # Get all video files
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(
                        os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

                    # Sort recordings by creation time (oldest first)
                    recordings.sort(key=os.path.getctime)

                    # Add numbering to the recordings
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))

                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )

            with gr.TabItem("📁 UI Configuration", id=8, visible=False):
                config_file_input = gr.File(
                    label="Load UI Settings from Config File",
                    file_types=[".json"],
                    interactive=True
                )
                with gr.Row():
                    load_config_button = gr.Button("Load Config", variant="primary")
                    save_config_button = gr.Button("Save UI Settings", variant="primary")

                config_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
                save_config_button.click(
                    fn=save_current_config,
                    inputs=[],  # 不需要输入参数
                    outputs=[config_status]
                )

            # New: Record tab
            with gr.TabItem("🛑 Record", id=9):
                
                gr.Markdown("### 🛑 Record User Input")
                with gr.Row():
                    with gr.Column(scale=2):
                        input_track_status = gr.Textbox(
                            label="Recording Status",
                            value="Recording not started",
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        input_track_start_btn = gr.Button("▶️ Start Recording", variant="primary")
                        input_track_stop_btn = gr.Button("⏹️ Stop Recording", variant="stop", interactive=False) # Stop initially not interactive

                gr.Markdown("### 📜 Last Recorded Trace Info") # New section for Record tab
                recorded_trace_info_display = gr.JSON(
                    label="Last Recorded Trace Details",
                    value={"message": "No trace recorded in this session yet."}
                )
                
                # Event handlers for Start/Stop recording buttons (remain in Record Tab)
                # Modified outputs for stop_input_tracking_with_context to include recorded_trace_info_display
                input_track_start_btn.click(
                    fn=start_input_tracking_with_context,
                    inputs=[],
                    # Outputs for start_input_tracking_with_context:
                    # 1. input_track_status Textbox
                    # 2. input_track_start_btn Button update
                    # 3. input_track_stop_btn Button update
                    # 4. trace_file_path Textbox (this will now be in Replay tab, but function still provides it)
                    outputs=[input_track_status, input_track_start_btn, input_track_stop_btn, trace_file_path] 
                )
                
                input_track_stop_btn.click(
                    fn=stop_input_tracking_with_context,
                    inputs=[],
                    # Outputs for stop_input_tracking_with_context:
                    # 1. input_track_status Textbox
                    # 2. input_track_start_btn Button update
                    # 3. input_track_stop_btn Button update
                    # 4. trace_file_path Textbox (this will now be in Replay tab, but function still provides it)
                    # 5. NEW: recorded_trace_info_display JSON
                    outputs=[input_track_status, input_track_start_btn, input_track_stop_btn, trace_file_path, recorded_trace_info_display]
                )


            # New: Replay Tab
            with gr.TabItem("▶️ Replay", id=10): # New Tab
                gr.Markdown("### 📂 Input Trace Files")
                
                refresh_traces_btn = gr.Button("🔄 Refresh Trace Files", variant="secondary")
                
                # The globally defined trace_file_path (now invisible) is used by handlers.
                # The display-only Textbox for the path that was here is now removed.

                trace_files_list = gr.Dataframe(
                    headers=["Name", "Created", "Size", "Events"],
                    label="Available Traces for Replay", # Clarified label
                    interactive=True,
                    wrap=True
                )
                
                with gr.Row():
                    trace_info_display_replay = gr.JSON( # Renamed to avoid conflict if original trace_info was different
                        label="Selected Trace File Info",
                        value={"message": "Select a trace file above to view details"}
                    )
                    trace_actions = gr.Column()
                    with trace_actions:
                        trace_replay_btn = gr.Button("▶️ Replay Selected Trace", variant="primary")
                        replay_speed_input = gr.Number(label="Replay Speed", value=2.0, minimum=0.1, interactive=True, scale=1)
                
                    # replay_status_output is now wrapped in its own Row
                with gr.Row(): 
                    replay_status_output = gr.Textbox(label="Replay Status", interactive=False)

                # Hidden state to store the full list of dicts from list_input_trace_files
                # This state is specific to the replay tab's list.
                trace_file_details_state_replay = gr.State([]) 

                # Event Handlers for Replay Tab
                # get_trace_file_path function definition (copied from original location)
                def get_trace_file_path(details_list: Optional[List[Dict[str, Any]]], evt: gr.SelectData) -> str:
                    logger.debug(f"--- DEBUG get_trace_file_path (Replay Tab): Event received: evt.index={evt.index}, evt.value='{evt.value}', evt.selected={evt.selected} ---")
                    if details_list is None:
                        logger.warning("--- DEBUG get_trace_file_path (Replay Tab): details_list is None. Returning empty. ---")
                        return ""
                    if not evt.index or not isinstance(evt.index, (list, tuple)) or len(evt.index) == 0:
                        logger.error(f"--- DEBUG get_trace_file_path (Replay Tab): evt.index is invalid: {evt.index}. Returning empty. ---")
                        return ""
                    potential_row_index = evt.index[0]
                    if not isinstance(potential_row_index, int):
                        logger.error(f"--- DEBUG get_trace_file_path (Replay Tab): Extracted potential_row_index '{potential_row_index}' is not an int. evt.index was: {evt.index}. Returning empty. ---")
                        return ""
                    row_index = potential_row_index
                    if 0 <= row_index < len(details_list):
                        selected_item_dict = details_list[row_index]
                        if isinstance(selected_item_dict, dict):
                            path_val = selected_item_dict.get("path")
                            logger.info(f"--- DEBUG get_trace_file_path (Replay Tab): Selected row index {row_index}, from details_list: {selected_item_dict}, extracted path: {path_val} ---")
                            return str(path_val) if path_val is not None else ""
                        else:
                            logger.warning(f"--- DEBUG get_trace_file_path (Replay Tab): Item at index {row_index} in details_list is not a dict: {selected_item_dict} (type: {type(selected_item_dict)}) ---")
                    else:
                        logger.warning(f"--- DEBUG get_trace_file_path (Replay Tab): Row index {row_index} out of bounds for details_list (0 to {len(details_list) -1}). ---")
                    return ""

                trace_files_list.select(
                    fn=get_trace_file_path,
                    inputs=[trace_file_details_state_replay], # Use the replay-specific state
                    outputs=[trace_file_path] # This updates the (now primarily) Replay tab's path display
                )
                
                # update_trace_info function definition (copied from original location)
                def update_trace_info_for_replay(local_trace_file_path: str): # Renamed parameter to avoid confusion
                    if not local_trace_file_path:
                        return {"message": "No trace file selected for replay"}
                    info = user_input_functions.get_file_info(local_trace_file_path)
                    return info
                
                trace_file_path.change( # This Textbox is now in the Replay tab
                    fn=update_trace_info_for_replay,
                    inputs=[trace_file_path],
                    outputs=[trace_info_display_replay] # Update the replay-specific JSON display
                )
                
                # refresh_traces function definition (copied from original location)
                def refresh_traces_for_replay(): # Renamed to be specific
                    try:
                        current_tracking_path = MANUAL_TRACES_DIR 
                        logger.debug(f"--- DEBUG webui.refresh_traces_for_replay: Path being used: '{current_tracking_path}' ---")
                        files = user_input_functions.list_input_trace_files(current_tracking_path)
                        rows = []
                        for i, file_dict_item in enumerate(files):
                            if not isinstance(file_dict_item, dict):
                                logger.warning(f"--- DEBUG webui.refresh_traces_for_replay: Item {i} is not a dict, skipping. ---")
                                continue
                            name_val = file_dict_item.get("name", "N/A") 
                            created_val = file_dict_item.get("created", "N/A")
                            size_val = file_dict_item.get("size", "N/A")
                            events_val = file_dict_item.get("events", "N/A")
                            if not isinstance(name_val, str):
                                logger.warning(f"--- DEBUG webui.refresh_traces_for_replay: 'name' field is not a string for item {i}: {name_val} (type: {type(name_val)}). Using 'Invalid Name'. ---")
                                name_val = "Invalid Name"
                            rows.append([name_val, created_val, size_val, events_val])
                        return rows, files # Returns for Dataframe and State
                    except Exception as e:
                        import traceback
                        logger.error(f"Fatal error in refresh_traces_for_replay: {str(e)}\\\\n{traceback.format_exc()}")
                        return ([["Error: " + str(e), "", "", ""]], [])
                        
                refresh_traces_btn.click( # This button is now in the Replay tab
                    fn=refresh_traces_for_replay,
                    inputs=[],
                    outputs=[trace_files_list, trace_file_details_state_replay] # Update replay-specific components
                )

                # replay_trace_wrapper function definition (copied from original location)
                # This function uses _global_browser_context which is managed by start/stop recording
                async def replay_trace_wrapper(local_trace_path_from_ui: str, local_replay_speed: float) -> str:
                    logger.info(f"--- DEBUG replay_trace_wrapper (Replay Tab): Called with path: {local_trace_path_from_ui}, Speed: {local_replay_speed} ---")
                    global _global_browser, _global_browser_context, _last_manual_trace_path 
                    # Context init logic (copied and adapted)
                    # ... (Assume context init logic from original replay_trace_wrapper is here)
                    # --- Start: Context Initialization Logic (adapted from start_input_tracking_with_context) ---
                    if _global_browser is None:
                        logger.info("--- DEBUG replay_trace_wrapper: Global browser not found, initializing for replay... ---")
                        _global_browser = CustomBrowser(
                            config=BrowserConfig(
                                headless=False, 
                                disable_security=True, 
                                cdp_url=os.getenv("CHROME_CDP_URL", os.getenv("CHROME_CDP", "http://localhost:9222")), # Prioritize CHROME_CDP_URL
                                chrome_instance_path=os.getenv("CHROME_PATH", None),
                                extra_chromium_args=[]
                            )
                        )
                        await _global_browser.async_init()
                    elif not (_global_browser.playwright_browser and _global_browser.playwright_browser.is_connected()):
                        logger.info("--- DEBUG replay_trace_wrapper: Global browser found but not connected. Re-initializing for replay... ---")
                        await _global_browser.async_init()

                    should_create_new_context = False
                    if _global_browser_context is None:
                        should_create_new_context = True
                    elif not _global_browser_context.playwright_context or not _global_browser_context.playwright_context.pages: # Check pages here
                        logger.info("--- DEBUG replay_trace_wrapper: Existing context has no pages or invalid Playwright link. Will create new one. ---")
                        await _global_browser_context.close() # Close unusable context
                        _global_browser_context = None
                        should_create_new_context = True
                    elif _global_browser_context.playwright_context and context_is_closed(_global_browser_context.playwright_context):
                        logger.info("--- DEBUG replay_trace_wrapper: Existing context connection closed. Will create new one. ---")
                        _global_browser_context = None # No need to explicitly close if already closed by Playwright
                        should_create_new_context = True
                    
                    if should_create_new_context:
                        logger.info(f"--- DEBUG replay_trace_wrapper: Attempting to initialize global browser context for replay. Current _global_browser.config.cdp_url: {_global_browser.config.cdp_url if _global_browser and _global_browser.config else 'N/A'} ---")
                        if _global_browser and _global_browser.config and _global_browser.config.cdp_url:
                            try:
                                logger.info(f"--- DEBUG replay_trace_wrapper: Reusing existing browser context via CDP: {_global_browser.config.cdp_url} ---")
                                _global_browser_context = await _global_browser.reuse_existing_context()
                                logger.info(f"--- DEBUG replay_trace_wrapper: Successfully reused existing browser context: {_global_browser_context} ---")
                                if _global_browser_context and _global_browser_context.playwright_context and not _global_browser_context.playwright_context.pages:
                                    logger.warning("--- DEBUG replay_trace_wrapper: Reused context has no pages. Opening a new blank tab in it for replay. ---")
                                    page = await _global_browser_context.playwright_context.new_page()
                                    await page.goto("about:blank") 
                                    await page.bring_to_front()
                            except Exception as e:
                                logger.error(f"--- DEBUG replay_trace_wrapper: Failed to reuse existing browser context: {e}. Falling back to new context strategy. ---")
                                _global_browser_context = None 
                        
                        if _global_browser_context is None: 
                            logger.info("--- DEBUG replay_trace_wrapper: Initializing new browser context for replay (not using CDP or reuse failed). ---")
                            if not (_global_browser and _global_browser.playwright):
                                logger.error("--- DEBUG replay_trace_wrapper: Global browser not available for creating new context. Cannot proceed with replay setup. ---")
                                return "Error: Browser not available for replay context."
                            
                            _global_browser_context = await _global_browser.new_context(
                                config=AppCustomBrowserContextConfig(
                                    enable_input_tracking=False, # Input tracking not needed for replay itself
                                    save_input_tracking_path="",   # Not saving new tracks during replay
                                    browser_window_size=BrowserContextWindowSize(width=1280, height=1100) 
                                )
                            )
                            if _global_browser_context and _global_browser_context.playwright_context:
                                try:
                                    page = await _global_browser_context.playwright_context.new_page()
                                    await page.bring_to_front()
                                    await page.goto("about:blank") 
                                    logger.info(f"--- DEBUG replay_trace_wrapper: New context and page ('{page.url if page else 'N/A'}') created for replay.")
                                except Exception as e:
                                    logger.error(f"--- DEBUG replay_trace_wrapper: Error opening new page for new replay context: {e}")
                                    return "Error: Could not prepare browser page for replay."
                            else:
                                logger.error("--- DEBUG replay_trace_wrapper: Failed to create a new browser context properly for replay.")
                                return "Error: Could not create browser context for replay."
                    # --- End: Context Initialization Logic ---

                    if not isinstance(_global_browser_context, CustomBrowserContext) or not _global_browser_context.pages:
                         logger.error(f"--- DEBUG replay_trace_wrapper: Context is not valid CustomBrowserContext or has no pages after init. Type: {type(_global_browser_context)}")
                         return "Error: Browser context is not ready for replay after setup."

                    logger.info(f"--- DEBUG replay_trace_wrapper: Setting browser context in user_input_functions. Context type: {type(_global_browser_context)} ---")
                    user_input_functions.set_browser_context(_global_browser_context)
                    
                    try:
                        success = await user_input_functions.replay_input_trace(local_trace_path_from_ui, speed=local_replay_speed)
                        if success:
                            status_message = "Input trace replay completed successfully."
                            _last_manual_trace_path = local_trace_path_from_ui 
                        else:
                            status_message = "Failed to replay input trace. See logs for details."
                        logger.info(f"--- DEBUG replay_trace_wrapper: Replay status: {status_message} ---")
                        return status_message
                    finally:
                        pass # Keep context open as per original logic in replay_input_trace

                trace_replay_btn.click( # This button is now in the Replay tab
                    fn=replay_trace_wrapper, 
                    inputs=[trace_file_path, replay_speed_input], 
                    outputs=[replay_status_output] # Output to the replay-specific status
                )
        
            # Attach the callback to the LLM provider dropdown
            llm_provider.change(
                lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
                inputs=[llm_provider, llm_api_key, llm_base_url],
                outputs=llm_model_name
            )

            # Add this after defining the components
            enable_recording.change(
                lambda enabled: gr.update(interactive=enabled),
                inputs=enable_recording,
                outputs=save_recording_path
            )

            use_own_browser.change(fn=close_global_browser)
            keep_browser_open.change(fn=close_global_browser)

            scan_and_register_components(demo)
            global webui_config_manager
            all_components = webui_config_manager.get_all_components()

            load_config_button.click(
                fn=update_ui_from_config,
                inputs=[config_file_input],
                outputs=all_components + [config_status]
            )
    return demo

# Build once; let the `gradio` CLI launch & reload
demo = create_ui(theme_name="Citrus")   # gradio looks for "demo"
app  = demo                            # optional alias, harmless

# --- allow plain `python webui.py` ----------------------------------
if __name__ == "__main__":              # executed only when you run: python webui.py
    # Ensure MANUAL_TRACES_DIR exists at startup
    Path(MANUAL_TRACES_DIR).mkdir(parents=True, exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True, allowed_paths=[MANUAL_TRACES_DIR])