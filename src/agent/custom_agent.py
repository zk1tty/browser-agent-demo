import json
import logging
import pdb
import traceback
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
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
from browser_use.browser.views import BrowserState
from browser_use.agent.prompts import PlannerPrompt

from json_repair import repair_json
from src.utils.agent_state import AgentState
from src.utils.replayer import TraceReplayer
from src.utils.user_input_tracker import UserInputTracker

from .custom_message_manager import CustomMessageManager, CustomMessageManagerSettings
from .custom_views import CustomAgentOutput, CustomAgentStepInfo, CustomAgentState as CustomAgentStateType

logger = logging.getLogger(__name__)

Context = TypeVar('Context')

# Define a simple structure for replay task details for clarity
class ReplayTaskDetails:
    def __init__(self, mode: str, trace_path: str, speed: float = 1.0, trace_save_path: Optional[str] = None):
        self.mode = mode
        self.trace_path = trace_path
        self.speed = speed
        self.trace_save_path = trace_save_path # For saving new traces if needed during an operation that might also record


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
            injected_agent_state: Optional[CustomAgentStateType] = None,
            context: Context | None = None,
    ):
        super().__init__(
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
            injected_agent_state=None,
            context=context,
        )
        # Initialize or restore CustomAgentState
        if injected_agent_state is not None and isinstance(injected_agent_state, CustomAgentStateType):
            self.state: CustomAgentStateType = injected_agent_state
        else:
            self.state: CustomAgentStateType = CustomAgentStateType()
            if injected_agent_state is not None: # Was provided but wrong type
                 logger.warning("injected_agent_state was provided but is not of type CustomAgentState. Initializing default CustomAgentState.")
        
        self.add_infos = add_infos
        # self.replay_event_file is removed, handled by task_input in run()

        self._message_manager = CustomMessageManager(
            task=self.task, # self.task is set by super().__init__
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
            state=self.state.message_manager_state, # Use state from CustomAgentStateType
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
        @dev : New Memory from LLM stores at important_contents.
            Usage of important_contents is
            - Track progress in repetitive tasks (e.g., "for each", "for all", "x times")
            - Store important information found during the task
            - Keep track of status and subresults for long tasks
            - Store extracted content from pages
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
        
        # NEW: Convert messages to serializable format without image_url
        def remove_image_url(item):
            """Helper function to remove image_url from a dictionary item"""
            if isinstance(item, dict):
                return {k: v for k, v in item.items() if k != 'image_url'}
            return item

        cleaned_messages = []
        for msg in fixed_input_messages:
            if isinstance(msg, dict):
                cleaned_messages.append(remove_image_url(msg))
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                # Handle LangChain message objects
                cleaned_msg = {
                    'type': msg.type,
                    'content': msg.content if not isinstance(msg.content, list) else [
                        remove_image_url(item) for item in msg.content
                    ]
                }
                cleaned_messages.append(cleaned_msg)
            else:
                # Handle other types
                cleaned_messages.append(str(msg))
        
        # Log both structure and content with JSON indentation
        # The final result is stored at debug/input_message.txt
        json_str = json.dumps([{'type': m.get('type') if isinstance(m, dict) else type(m).__name__, 'content': m.get('content') if isinstance(m, dict) else m} for m in cleaned_messages], indent=2)
        formatted_str = json_str.replace(r'\n', '\n')
        logger.debug(f"AI_input_messages: {formatted_str}")

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
        if not hasattr(self.state, 'history') or not self.state.history:
            return "No browsing history yet."
        
        summary_lines = []
        try:
            # Iterate backwards through history items
            for history_item in reversed(self.state.history.history):
                if len(summary_lines) >= max_steps:
                    break
                
                page_title = history_item.state.page_title if history_item.state else "Unknown Page"
                url = history_item.state.url if history_item.state else "Unknown URL"
                
                actions_summary = []
                if history_item.action:
                    for action_detail in history_item.action:
                        # action_detail is ActionDetail, which has .action (BaseNamedAction)
                        if action_detail.action:
                            action_str = f"{action_detail.action.name}"
                            # Add arguments if any, simplified
                            args_str = json.dumps(action_detail.action.arguments) if action_detail.action.arguments else ""
                            if args_str and args_str !="{}":
                                action_str += f"({args_str})"
                            actions_summary.append(action_str)
                
                action_desc = "; ".join(actions_summary) if actions_summary else "No action taken"
                summary_line = f"- Step {history_item.metadata.step_number}: [{page_title}]({url}) - Action: {action_desc}\\n"
                
                if sum(len(s) for s in summary_lines) + len(summary_line) > max_chars and summary_lines:
                    summary_lines.append("... (history truncated due to length)")
                    break
                summary_lines.append(summary_line)
        except Exception as e:
            logger.error(f"Error summarizing browsing history: {e}")
            return "Error summarizing history."

        if not summary_lines:
            return "No actions recorded in recent history."
        return "Browsing History (Recent Steps):\\n" + "".join(reversed(summary_lines))

    @time_execution_async("--step")
    async def step(self, step_info: Optional[CustomAgentStepInfo] = None) -> None:
        if not step_info: # Should be initialized in run loop
            logger.error("step_info not provided to CustomAgent.step")
            return

        model_output = None # Initialize to ensure it's defined for finally
        state = None # Initialize
        result = None # Initialize
        tokens = 0 # Initialize
        step_start_time = time.time()

        try:
            state = await self.browser_context.get_state()
            await self._raise_if_stopped_or_paused()

            history_summary_str = self._summarize_browsing_history(max_steps=5, max_chars=1500)

            self.message_manager.add_state_message(
                state,
                self.state.last_action,
                self.state.last_result,
                step_info,
                self.settings.use_vision,
                history_summary=history_summary_str
            )

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
                    self.message_manager._remove_state_message_by_index(-1)
                await self._raise_if_stopped_or_paused()
            except Exception as e:
                self.message_manager._remove_state_message_by_index(-1)
                raise e

            result = await self.multi_act(model_output.action)
            for ret_ in result:
                if ret_.extracted_content and "Extracted page" in ret_.extracted_content:
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
            actions_telemetry = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output and hasattr(model_output, 'action') and model_output.action else []
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps,
                    actions=actions_telemetry,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=[r.error for r in result if r.error] if result else ['No result'],
                )
            )
            if not result:
                return

            if state and model_output:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)

    # New: modified to accept ReplayTaskDetails at replay mode
    async def run(self, task_input: Union[str, ReplayTaskDetails], max_steps: int = 100) -> Optional[AgentHistoryList]:
        if isinstance(task_input, ReplayTaskDetails) and task_input.mode == "replay":
            logger.info(f"Starting replay mode from trace: {task_input.trace_path} with speed: {task_input.speed}")
            if not self.browser_context:
                logger.error("Browser context not available for replay.")
                return None

            current_page = self.page # self.page is set in Agent.__init__ from browser_context.pages[0]
            if not current_page or current_page.is_closed():
                logger.info("Agent's default page is not available or closed. Trying to get a page from context.")
                if self.browser_context.pages:
                    for p in self.browser_context.pages: # Find first open page
                        if not p.is_closed(): current_page = p; break 
                    if not current_page or current_page.is_closed(): # If all are closed or none found initially
                        logger.info("No open pages found in context or all were closed. Creating a new page for replay.")
                        current_page = await self.browser_context.new_page()
                else:
                    logger.info("No pages in context. Creating a new page for replay.")
                    current_page = await self.browser_context.new_page()
                self.page = current_page # Update self.page to the new/active page

            if not self.browser or not current_page: # self.browser should exist if browser_context does
                logger.error("Browser or page not available for replay.")
                return None
            
            rep = TraceReplayer(self.browser, current_page)
            try:
                await rep.play(task_input.trace_path, speed=task_input.speed)
                logger.info(f"Replay finished for trace: {task_input.trace_path}")
            except FileNotFoundError:
                logger.error(f"Trace file not found: {task_input.trace_path}")
            except Exception as e:
                logger.error(f"Error during replay of {task_input.trace_path}: {e}")
                traceback.print_exc() # Log full traceback for replay errors
            return None # Replay mode does not return AgentHistoryList

        else: # Autonomous mode
            original_task_description = self.task
            if isinstance(task_input, str) and self.task != task_input:
                logger.info(f"Autonomous run: Task updated from '{self.task}' to '{task_input}'")
                self.task = task_input
                self._message_manager.task = self.task # Update message manager's task
                 # Reset or update initial messages in message manager if task significantly changes
                self._message_manager.state.history.clear() # Clear previous messages for new task
                self._message_manager.add_initial_messages(self.task, self.add_infos)
            elif not isinstance(task_input, str):
                 logger.warning(f"Autonomous run: task_input is not a string ({type(task_input)}). Using existing task: {self.task}")


            logger.info(f"Starting autonomous agent run for task: '{self.task}', max_steps: {max_steps}")
            
            # Use the base Agent.run() method for the main loop and its own try/finally for telemetry etc.
            history: Optional[AgentHistoryList] = await super().run(max_steps=max_steps)

            # After autonomous run, persist UserInputTracker history
            if self.browser_context and hasattr(self.browser_context, 'input_tracker') and self.browser_context.input_tracker:
                tracker: UserInputTracker = self.browser_context.input_tracker
                
                if tracker.events:
                    try:
                        trace_content = tracker.export_events_to_jsonl()
                        
                        # Determine save path
                        trace_dir = "./tmp/traces" # TODO: Make this configurable
                        os.makedirs(trace_dir, exist_ok=True)
                        
                        # Default filename with timestamp
                        default_trace_filename = f"session_{time.strftime('%Y%m%d_%H%M%S')}.trace"
                        session_trace_path = os.path.join(trace_dir, default_trace_filename)

                        # Allow overriding save path via task_input if it's ReplayTaskDetails (though unlikely for saving *after* autonomous)
                        # Or if task_input was a dict with this key (more general for future task structures)
                        final_save_path = session_trace_path
                        if isinstance(task_input, ReplayTaskDetails) and task_input.trace_save_path:
                             final_save_path = task_input.trace_save_path
                        elif isinstance(task_input, dict) and task_input.get('trace_save_path'):
                             final_save_path = task_input['trace_save_path']
                        
                        with open(final_save_path, "w") as f:
                            f.write(trace_content)
                        logger.info(f"User input trace saved to {final_save_path}")
                        
                        # Optional: Clear events after saving to prevent re-saving or growing memory indefinitely
                        # tracker.events.clear() 
                        # logger.info("Cleared tracker events after saving.")

                    except Exception as e:
                        logger.error(f"Failed to save user input trace: {e}")
                        traceback.print_exc()
                elif tracker.is_recording:
                     logger.info("User input tracker was active, but no events were recorded to save.")
                else:
                    logger.info("User input tracker was not active or no events to save.")
            else:
                logger.warning("UserInputTracker not found on browser_context or not initialized. Cannot save trace.")
            
            return history
