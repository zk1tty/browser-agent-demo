import json
import logging
import pdb
import traceback
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Type, TypeVar
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io
import asyncio
import time
import platform
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.service import Agent
from browser_use.agent.message_manager.utils import convert_input_messages, extract_json_from_model_output, \
    save_conversation
from browser_use.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentState,
    AgentStepInfo,
    StepMetadata,
    ToolCallingMethod,
)
from browser_use.agent.gif import create_history_gif
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserStateHistory
from browser_use.controller.service import Controller
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.agent.prompts import PlannerPrompt

from json_repair import repair_json
from src.utils.agent_state import AgentState

from .custom_message_manager import CustomMessageManager, CustomMessageManagerSettings
from .custom_views import CustomAgentOutput, CustomAgentStepInfo, CustomAgentState

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class CustomAgent(Agent):
    def __init__(
            self,
            task: str,
            llm: BaseChatModel,
            add_infos: str = "",
            # Optional parameters
            browser: Browser | None = None,
            browser_context: BrowserContext | None = None,
            controller: Controller[Context] = Controller(),
            # Initial agent run parameters
            sensitive_data: Optional[Dict[str, str]] = None,
            initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
            # Cloud Callbacks
            register_new_step_callback: Callable[['BrowserState', 'AgentOutput', int], Awaitable[None]] | None = None,
            register_done_callback: Callable[['AgentHistoryList'], Awaitable[None]] | None = None,
            register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
            # Agent settings
            use_vision: bool = True,
            use_vision_for_planner: bool = False,
            save_conversation_path: Optional[str] = None,
            save_conversation_path_encoding: Optional[str] = 'utf-8',
            max_failures: int = 3,
            retry_delay: int = 10,
            system_prompt_class: Type[SystemPrompt] = SystemPrompt,
            agent_prompt_class: Type[AgentMessagePrompt] = AgentMessagePrompt,
            max_input_tokens: int = 128000,
            validate_output: bool = False,
            message_context: Optional[str] = None,
            generate_gif: bool | str = False,
            available_file_paths: Optional[list[str]] = None,
            include_attributes: list[str] = [
                'title',
                'type',
                'name',
                'role',
                'aria-label',
                'placeholder',
                'value',
                'alt',
                'aria-expanded',
                'data-date-format',
            ],
            max_actions_per_step: int = 10,
            tool_calling_method: Optional[ToolCallingMethod] = 'auto',
            page_extraction_llm: Optional[BaseChatModel] = None,
            planner_llm: Optional[BaseChatModel] = None,
            planner_interval: int = 1,  # Run planner every N steps
            # Inject state
            injected_agent_state: Optional[AgentState] = None,
            context: Context | None = None,
    ):
        super(CustomAgent, self).__init__(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            sensitive_data=sensitive_data,
            initial_actions=initial_actions,
            register_new_step_callback=register_new_step_callback,
            register_done_callback=register_done_callback,
            register_external_agent_status_raise_error_callback=register_external_agent_status_raise_error_callback,
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            page_extraction_llm=page_extraction_llm,
            planner_llm=planner_llm,
            planner_interval=planner_interval,
            injected_agent_state=injected_agent_state,
            context=context,
        )
        self.state = injected_agent_state or CustomAgentState()
        self.add_infos = add_infos
        self._message_manager = CustomMessageManager(
            task=task,
            system_message=self.settings.system_prompt_class(
                self.available_actions,
                max_actions_per_step=self.settings.max_actions_per_step,
            ).get_system_message(),
            settings=CustomMessageManagerSettings(
                max_input_tokens=self.settings.max_input_tokens,
                include_attributes=self.settings.include_attributes,
                message_context=self.settings.message_context,
                sensitive_data=sensitive_data,
                available_file_paths=self.settings.available_file_paths,
                agent_prompt_class=agent_prompt_class
            ),
            state=self.state.message_manager_state,
        )

    ## TODO: Eval the response from LLM
    def _log_response(self, response: CustomAgentOutput) -> None:
        """Log the model's response"""
        if "Success" in response.current_state.evaluation_previous_goal:
            emoji = "✅"
        elif "Failed" in response.current_state.evaluation_previous_goal:
            emoji = "❌"
        else:
            emoji = "🤷"

        logger.info(f"{emoji} Eval: {response.current_state.evaluation_previous_goal}")
        logger.info(f"🧠 New Memory: {response.current_state.important_contents}")
        logger.info(f"🤔 Thought: {response.current_state.thought}")
        logger.info(f"🎯 Next Goal: {response.current_state.next_goal}")
        for i, action in enumerate(response.action):
            logger.info(
                f"🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}"
            )

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = CustomAgentOutput.type_with_custom_actions(self.ActionModel)

    def update_step_info(
            self, model_output: CustomAgentOutput, step_info: CustomAgentStepInfo = None
    ):
        """
        update step info
        """
        if step_info is None:
            return

        step_info.step_number += 1
        important_contents = model_output.current_state.important_contents
        if (
                important_contents
                and "None" not in important_contents
                and important_contents not in step_info.memory
        ):
            step_info.memory += important_contents + "\n"

        logger.info(f"🧠 All Memory: \n{step_info.memory}")

    # hint: get next action from LLM by calling llm.invoke in utils/llm.py
    @time_execution_async("--get_next_action")
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state"""
        fixed_input_messages = self._convert_input_messages(input_messages)
        
        # Added: Convert messages to serializable format
        cleaned_messages = []
        for msg in fixed_input_messages:
            if isinstance(msg, dict):
                # Create a copy of the message without image_url
                cleaned_msg = {k: v for k, v in msg.items() if k != 'image_url'}
                cleaned_messages.append(cleaned_msg)
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                # Handle LangChain message objects
                cleaned_msg = {
                    'type': msg.type,
                    'content': msg.content if not isinstance(msg.content, list) else [
                        {k: v for k, v in item.items() if k != 'image_url'}
                        for item in msg.content
                    ]
                }
                cleaned_messages.append(cleaned_msg)
            else:
                # Handle other types
                cleaned_messages.append(str(msg))
        
        logger.debug(f"fixed_input_messages: {json.dumps(cleaned_messages, indent=2)}")

        # TODO: This is where the LLM is called
        ai_message = self.llm.invoke(fixed_input_messages)
        self.message_manager._add_message_with_tokens(ai_message)

        if hasattr(ai_message, "reasoning_content"):
            logger.info("🤯 Start Deep Thinking: ")
            logger.info(ai_message.reasoning_content)
            logger.info("🤯 End Deep Thinking")

        if isinstance(ai_message.content, list):
            ai_content = ai_message.content[0]
        else:
            ai_content = ai_message.content

        try:
            ai_content = ai_content.replace("```json", "").replace("```", "")
            ai_content = repair_json(ai_content)
            parsed_json = json.loads(ai_content)
            
            # Debug log the parsed JSON
            logger.debug(f"===Parsed JSON bf mod===: {json.dumps(parsed_json, indent=2)}")
            
            # Ensure parsed_json is a dictionary
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Expected dictionary but got {type(parsed_json)}")
            
            # Handle switch_tab action by ensuring page_id is present
            if 'action' in parsed_json and isinstance(parsed_json['action'], list):
                for action in parsed_json['action']:
                    if action.get('type') == 'switch_tab':
                        # Handle both 'index' and 'tab_index' cases
                        tab_index = action.get('tab_index') or action.get('index')
                        if tab_index is not None:
                            # Get current browser state
                            state = await self.browser_context.get_state()
                            if state and hasattr(state, 'pages') and len(state.pages) > tab_index:
                                # Get the page_id from the browser state
                                action['page_id'] = state.pages[tab_index].page_id
                                # Ensure tab_index is used consistently
                                action['tab_index'] = tab_index
                                if 'index' in action:
                                    del action['index']
                            else:
                                raise ValueError(f"Invalid tab_index/index: {tab_index}")
                        else:
                            raise ValueError("Missing tab_index or index in switch_tab action")
            
            # Debug log the modified JSON
            logger.debug(f"===Parsed JSON af mod===:: {json.dumps(parsed_json, indent=2)}")
            
            # Ensure current_state is present and properly structured
            if 'current_state' not in parsed_json:
                raise ValueError("Missing 'current_state' in response")
            
            if not isinstance(parsed_json['current_state'], dict):
                raise ValueError(f"Expected dictionary for current_state but got {type(parsed_json['current_state'])}")
            
            # Create the CustomAgentOutput instance
            parsed: AgentOutput = self.AgentOutput(**parsed_json)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.debug(f"Error parsing response. Content: {ai_message.content}")
            raise ValueError('Could not parse response.')

        if parsed is None:
            logger.debug(ai_message.content)
            raise ValueError('Could not parse response.')

        # cut the number of actions to max_actions_per_step if needed
        if len(parsed.action) > self.settings.max_actions_per_step:
            parsed.action = parsed.action[: self.settings.max_actions_per_step]
        self._log_response(parsed)
        return parsed

    async def _run_planner(self) -> Optional[str]:
        """Run the planner to analyze state and suggest next steps"""
        # Skip planning if no planner_llm is set
        if not self.settings.planner_llm:
            return None

        # Create planner message history using full message history
        planner_messages = [
            PlannerPrompt(self.controller.registry.get_prompt_description()).get_system_message(),
            *self.message_manager.get_messages()[1:],  # Use full message history except the first
        ]

        if not self.settings.use_vision_for_planner and self.settings.use_vision:
            last_state_message: HumanMessage = planner_messages[-1]
            # remove image from last state message
            new_msg = ''
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg['type'] == 'text':
                        new_msg += msg['text']
                    elif msg['type'] == 'image_url':
                        continue
            else:
                new_msg = last_state_message.content

            planner_messages[-1] = HumanMessage(content=new_msg)

        # Get planner output
        response = await self.settings.planner_llm.ainvoke(planner_messages)
        plan = str(response.content)
        # console log plan
        print(f"plan: {plan}")
        last_state_message = self.message_manager.get_messages()[-1]
        if isinstance(last_state_message, HumanMessage):
            # remove image from last state message
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg['type'] == 'text':
                        msg['text'] += f"\nPlanning Agent outputs plans:\n {plan}\n"
            else:
                last_state_message.content += f"\nPlanning Agent outputs plans:\n {plan}\n "

        try:
            plan_json = json.loads(plan.replace("```json", "").replace("```", ""))
            logger.info(f'📋 Plans:\n{json.dumps(plan_json, indent=4)}')

            if hasattr(response, "reasoning_content"):
                logger.info("🤯 Start Planning Deep Thinking: ")
                logger.info(response.reasoning_content)
                logger.info("🤯 End Planning Deep Thinking")

        except json.JSONDecodeError:
            logger.info(f'📋 Plans:\n{plan}')
        except Exception as e:
            logger.debug(f'Error parsing planning analysis: {e}')
            logger.info(f'📋 Plans: {plan}')
        return plan

    def _summarize_browsing_history(self, max_steps: int = 5, max_chars: int = 1500) -> str:
        if not self.state.history or not self.state.history.history:
            return "No browsing history yet."

        summary_lines = []
        char_count = 0

        # Iterate backwards through history (most recent first)
        for history_item in reversed(self.state.history.history):
            if len(summary_lines) >= max_steps:
                break

            step_num = history_item.metadata.step_number
            url = history_item.state.url
            # Get title from state or first tab
            title = getattr(history_item.state, 'title', '')
            if hasattr(history_item.state, 'tabs') and history_item.state.tabs:
                first_tab = history_item.state.tabs[0]
                tab_title = getattr(first_tab, 'title', '')
                if tab_title:
                    title = tab_title

            actions_summary = []
            errors_summary = []

            if history_item.result:
                for res in history_item.result:
                    # Use model_output action description if available and parsed
                    if history_item.model_output and history_item.model_output.action:
                         # Simplistic representation, might need more detail
                        action_type = getattr(history_item.model_output.action, 'action_type', 'unknown')
                        action_args = getattr(history_item.model_output.action, 'args', {})
                        action_desc = f"{action_type}({action_args})"
                        actions_summary.append(action_desc[:100]) # Truncate
                    # Fallback or augment with extracted content if no model_output action
                    elif res.extracted_content and not res.error:
                        action_desc = res.extracted_content.split('\\n')[0] # Take first line
                        actions_summary.append(action_desc[:100]) # Truncate

                    if res.error:
                        # Summarize error - Get the most specific part of the error
                        error_lines = res.error.strip().split('\\n')
                        error_line = error_lines[-1] if error_lines else "Unknown Error"
                        errors_summary.append(f"Error: ...{error_line[-150:]}") # Truncate

            # Deduplicate action summaries if they come from both model_output and extracted_content
            actions_summary = list(dict.fromkeys(actions_summary))

            line = f"Step {step_num}: URL: {url}"
            if title:
                 line += f" (Title: {title[:50]}...)" # Truncate title
            if actions_summary:
                line += f" | Actions: {'; '.join(actions_summary)}"
            if errors_summary:
                line += f" | Results: {'; '.join(errors_summary)}"
            elif history_item.result: # Check if there were results at all
                 line += " | Results: OK" # Assume OK if results exist but no errors were logged

            line += "\\n"

            if char_count + len(line) > max_chars and summary_lines:
                 # Stop if adding this line exceeds char limit (and we already have some lines)
                 summary_lines.append("... (history truncated due to length)\\n")
                 break

            summary_lines.append(line)
            char_count += len(line)

        if not summary_lines:
            return "No recent history processed."

        # Reverse back to chronological order and join
        return "Browsing History (Recent Steps):\\n" + "".join(reversed(summary_lines))

    @time_execution_async("--step")
    async def step(self, step_info: Optional[CustomAgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        logger.info(f"\n📍 Step {self.state.n_steps}")
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_start_time = time.time()
        tokens = 0

        try:
            state = await self.browser_context.get_state()
            await self._raise_if_stopped_or_paused()

            # Generate history summary before adding state message
            history_summary_str = self._summarize_browsing_history(max_steps=5, max_chars=1500)

            # Pass the summary to add_state_message
            self.message_manager.add_state_message(
                state,
                self.state.last_action,
                self.state.last_result,
                step_info,
                self.settings.use_vision,
                history_summary=history_summary_str # Pass the summary
            )

            # Run planner at specified intervals if planner is configured
            if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
                await self._run_planner()
            input_messages = self.message_manager.get_messages()
            tokens = self._message_manager.state.history.current_tokens

            try:
                model_output = await self.get_next_action(input_messages)
                self.update_step_info(model_output, step_info)
                self.state.n_steps += 1

                if self.register_new_step_callback:
                    await self.register_new_step_callback(state, model_output, self.state.n_steps)

                if self.settings.save_conversation_path:
                    target = self.settings.save_conversation_path + f'_{self.state.n_steps}.txt'
                    save_conversation(input_messages, model_output, target,
                                      self.settings.save_conversation_path_encoding)

                if self.model_name != "deepseek-reasoner":
                    # remove prev message
                    self.message_manager._remove_state_message_by_index(-1)
                await self._raise_if_stopped_or_paused()
            except Exception as e:
                # model call failed, remove last state message from history
                self.message_manager._remove_state_message_by_index(-1)
                raise e

            result: list[ActionResult] = await self.multi_act(model_output.action)
            for ret_ in result:
                if ret_.extracted_content and "Extracted page" in ret_.extracted_content:
                    # record every extracted page
                    if ret_.extracted_content[:100] not in self.state.extracted_content:
                        self.state.extracted_content += ret_.extracted_content
            self.state.last_result = result
            self.state.last_action = model_output.action
            if len(result) > 0 and result[-1].is_done:
                if not self.state.extracted_content:
                    self.state.extracted_content = step_info.memory
                result[-1].extracted_content = self.state.extracted_content
                logger.info(f"📄 Result: {result[-1].extracted_content}")

            self.state.consecutive_failures = 0

        except InterruptedError:
            logger.debug('Agent paused')
            self.state.last_result = [
                ActionResult(
                    error='The agent was paused - now continuing actions might need to be repeated',
                    include_in_memory=True
                )
            ]
            return

        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result

        finally:
            step_end_time = time.time()
            actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps,
                    actions=actions,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=[r.error for r in result if r.error] if result else ['No result'],
                )
            )
            if not result:
                return

            if state:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""
        try:
            self._log_agent_run()

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            step_info = CustomAgentStepInfo(
                task=self.task,
                add_infos=self.add_infos,
                step_number=1,
                max_steps=max_steps,
                memory="",
            )

            for step in range(max_steps):
                # Check if we should stop due to too many failures
                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    break

                # Check control flags before each step
                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
                    if self.state.stopped:  # Allow stopping while paused
                        break

                await self.step(step_info)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    break
            else:
                logger.info("❌ Failed to complete task in maximum steps")
                if not self.state.extracted_content:
                    self.state.history.history[-1].result[-1].extracted_content = step_info.memory
                else:
                    self.state.history.history[-1].result[-1].extracted_content = self.state.extracted_content

            return self.state.history

        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.state.agent_id,
                    is_done=self.state.history.is_done(),
                    success=self.state.history.is_successful(),
                    steps=self.state.n_steps,
                    max_steps_reached=self.state.n_steps >= max_steps,
                    errors=self.state.history.errors(),
                    total_input_tokens=self.state.history.total_input_tokens(),
                    total_duration_seconds=self.state.history.total_duration_seconds(),
                )
            )

            if not self.injected_browser_context:
                await self.browser_context.close()

            if not self.injected_browser and self.browser:
                await self.browser.close()

            if self.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif

                create_history_gif(task=self.task, history=self.state.history, output_path=output_path)
