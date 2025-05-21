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

from pydantic import BaseModel
from json_repair import repair_json
from src.utils.agent_state import AgentState
from src.utils.replayer import TraceReplayer, load_trace, Drift
from src.utils.user_input_tracker import UserInputTracker

from .custom_message_manager import CustomMessageManager, CustomMessageManagerSettings
from .custom_views import CustomAgentOutput, CustomAgentStepInfo, CustomAgentState as CustomAgentStateType, CustomAgentBrain

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
        self.current_task_memory: str = "" # Initialize custom memory

        self._message_manager: CustomMessageManager = CustomMessageManager(
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
            self, model_output: CustomAgentOutput, step_info: Optional[CustomAgentStepInfo] = None
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
    async def get_next_action(self, input_messages: list[BaseMessage]) -> CustomAgentOutput:
        """Get next action from LLM based on current state"""
        
        # The _convert_input_messages and cleaned_messages logic seems to have been
        # for a specific format possibly expected by a previous _get_model_output method.
        # We will now directly use self.llm.ainvoke with input_messages (List[BaseMessage]).
        # The logic for removing image_urls, if still needed, would have to be 
        # applied to input_messages before this call, or handled by the LLM itself.

        if not self.llm:
            logger.error("LLM not initialized in CustomAgent.")
            # Return an error structure that _parse_model_output can handle
            # This assumes _parse_model_output can parse a JSON string error.
            # The actual error handling might need to be more robust based on _parse_model_output's capabilities.
            # Also, self.AgentOutput needs to be available here.
            if not hasattr(self, 'AgentOutput') or not self.AgentOutput:
                self._setup_action_models() # Ensure AgentOutput is set up
            
            # Construct a raw string that _parse_model_output can work with to produce an AgentOutput
            # This usually involves a JSON string that looks like what the LLM would output in an error case.
            # For now, an empty actions list and an error message in thought/state might be a way.
            # This is a placeholder for robust error generation.
            error_payload = {
                "current_state": {
                    "evaluation_previous_goal": "Error",
                    "important_contents": "LLM not initialized.",
                    "thought": "Critical error: LLM not initialized.",
                    "next_goal": "Cannot proceed."
                },
                "action": []
            }
            model_output_raw = json.dumps(error_payload)
            return self._parse_model_output(model_output_raw, self.ActionModel)

        try:
            llm_response = await self.llm.ainvoke(input_messages)
            
            # model_output_raw should be a string, typically the content from the LLM response.
            # The base class's _parse_model_output is expected to handle this string.
            if hasattr(llm_response, 'content') and llm_response.content is not None:
                model_output_raw = str(llm_response.content)
            elif isinstance(llm_response, AIMessage) and llm_response.tool_calls:
                # If content is None but there are tool_calls, the parser might expect
                # a specific string format (e.g., JSON of tool_calls) or to handle AIMessage directly.
                # Forcing it to string for now, assuming the parser can handle stringified tool_calls
                # or that the main information is in .content and tool_calls are metadata for the parser.
                # This part is sensitive to how the base Agent's parser works.
                # A common robust approach is for the LLM to put tool call JSON into the .content string.
                # If not, serializing tool_calls to JSON is a common fallback if the parser expects it.
                try:
                    # Attempt to create a JSON string that might represent the tool calls
                    # ToolCall objects in Langchain are typically TypedDicts and directly serializable.
                    model_output_raw = json.dumps(llm_response.tool_calls)
                except Exception as serialization_error:
                    logger.warning(f"Could not serialize tool_calls for AIMessage: {serialization_error}. Falling back to str(AIMessage).")
                    model_output_raw = str(llm_response) # Fallback to full string representation
            else:
                model_output_raw = str(llm_response) # General fallback

        except Exception as e:
            logger.error(f"Error invoking LLM: {e}", exc_info=True)
            error_payload = {
                "current_state": {
                    "evaluation_previous_goal": "Error",
                    "important_contents": f"LLM invocation error: {str(e)}",
                    "thought": f"LLM invocation error: {str(e)}",
                    "next_goal": "Cannot proceed."
                },
                "action": []
            }
            model_output_raw = json.dumps(error_payload)

        # Parse the model output
        # Ensure self.ActionModel is available for the parser
        if not hasattr(self, 'ActionModel') or not self.ActionModel:
            self._setup_action_models() # Ensure ActionModel is set up for parsing

        parsed_output = self._parse_model_output(model_output_raw, self.ActionModel)
        return parsed_output

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
            # Type hint for last_state_message was HumanMessage, ensure planner_messages[-1] is HumanMessage or check type
            last_planner_message = planner_messages[-1]
            new_msg_content: Union[str, List[Dict[str, Any]]] = '' # type for new content
            
            if isinstance(last_planner_message, HumanMessage):
                if isinstance(last_planner_message.content, list):
                    processed_content_list = []
                    for item in last_planner_message.content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                processed_content_list.append({'type': 'text', 'text': item.get('text', '')})
                            # Keep other dict types if necessary, or filter image_url
                            elif item.get('type') == 'image_url':
                                continue # Skip image
                            else:
                                processed_content_list.append(item) # Keep other dicts
                        elif isinstance(item, str):
                            processed_content_list.append({'type': 'text', 'text': item}) # Convert str to dict
                    new_msg_content = processed_content_list
                    # Reconstruct new_msg from processed_content_list if needed as a single string
                    temp_new_msg = ""
                    for item_content in new_msg_content: # new_msg_content is List[Dict[str,Any]]
                        if isinstance(item_content, dict) and item_content.get('type') == 'text':
                             temp_new_msg += item_content.get('text','')
                    new_msg = temp_new_msg

                elif isinstance(last_planner_message.content, str):
                    new_msg = last_planner_message.content
                
                planner_messages[-1] = HumanMessage(content=new_msg if new_msg else last_planner_message.content)


        # Get planner output
        response = await self.settings.planner_llm.ainvoke(planner_messages)
        plan = str(response.content)
        # console log plan
        print(f"plan: {plan}")
        
        last_message_from_manager = self.message_manager.get_messages()[-1]
        if isinstance(last_message_from_manager, HumanMessage):
            # Target last_message_from_manager (which is a HumanMessage) for modification
            if isinstance(last_message_from_manager.content, list):
                # Create a new list for content to avoid modifying immutable parts if any
                new_content_list = []
                modified = False
                for item in last_message_from_manager.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        current_text = item.get('text', '')
                        # Create a new dict for the modified text item
                        new_content_list.append({'type': 'text', 'text': current_text + f"\\nPlanning Agent outputs plans:\\n {plan}\\n"})
                        modified = True
                    else:
                        new_content_list.append(item) # Keep other items as is
                if modified:
                    last_message_from_manager.content = new_content_list
                else: # If no text item was found to append to, add a new one
                    new_content_list.append({'type': 'text', 'text': f"\\nPlanning Agent outputs plans:\\n {plan}\\n"})
                    last_message_from_manager.content = new_content_list

            elif isinstance(last_message_from_manager.content, str):
                last_message_from_manager.content += f"\\nPlanning Agent outputs plans:\\n {plan}\\n "
            # If no modification happened (e.g. content was not list or str, or list had no text part)
            # one might consider appending a new HumanMessage with the plan, but that changes history structure.

        try:
            plan_json = json.loads(plan.replace("```json", "").replace("```", ""))
            logger.info(f'📋 Plans:\\n{json.dumps(plan_json, indent=4)}')

            reasoning_content = getattr(response, "reasoning_content", None)
            if reasoning_content:
                logger.info("🤯 Start Planning Deep Thinking: ")
                logger.info(reasoning_content)
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
                
                page_title = getattr(history_item.state, "page_title", "Unknown Page") if history_item.state else "Unknown Page"
                url = getattr(history_item.state, "url", "Unknown URL") if history_item.state else "Unknown URL"
                
                actions_summary = []
                current_actions = history_item.model_output.action if history_item.model_output and hasattr(history_item.model_output, 'action') else []
                if current_actions:
                    for act_model in current_actions: # act_model is ActionModel
                        if hasattr(act_model, 'name'):
                            action_str = f"{act_model.name}" # type: ignore[attr-defined]
                            args_str = json.dumps(act_model.arguments) if hasattr(act_model, 'arguments') and act_model.arguments else "" # type: ignore[attr-defined]
                            if args_str and args_str !="{}":
                                action_str += f"({args_str})"
                            actions_summary.append(action_str)
                
                action_desc = "; ".join(actions_summary) if actions_summary else "No action taken"
                step_num_str = f"Step {history_item.metadata.step_number}" if history_item.metadata and hasattr(history_item.metadata, 'step_number') else "Step Unknown"
                summary_line = f"- {step_num_str}: [{page_title}]({url}) - Action: {action_desc}\\\\n"
                
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
    async def step(self, base_step_info: Optional[AgentStepInfo] = None) -> None:
        # The base_step_info comes from the superclass Agent's run loop.
        # We need to create a CustomAgentStepInfo for our custom prompts.
        
        # if not base_step_info: # This check might be too strict if super().run() doesn't always provide it.
        #     logger.error("base_step_info not provided to CustomAgent.step by superclass run loop.")
        #     # Decide how to handle this: error out, or create a default?
        #     # For now, let's assume it's provided or self.state is the source of truth for step numbers.
        #     # If super().run() manages step counts, base_step_info.step_number would be relevant.
        #     # If CustomAgent manages its own (self.state.n_steps), use that.
        #     # Let's use self.state for step counts as it seems to be incremented by CustomAgent.
        
        current_custom_step_info = CustomAgentStepInfo(
            step_number=self.state.n_steps,  # Use self.state.n_steps
            max_steps=self.state.max_steps if self.state.max_steps is not None else 100, # Get from state or default
            task=self.task,
            add_infos=self.add_infos,
            memory=self.current_task_memory
        )

        model_output = None # Initialize to ensure it's defined for finally
        state = None # Initialize
        result = None # Initialize
        tokens = 0 # Initialize
        step_start_time = time.time()

        try:
            logger.debug("CustomAgent.step: About to call self.browser_context.get_state()")
            state = await self.browser_context.get_state()
            logger.debug(f"CustomAgent.step: self.browser_context.get_state() returned. URL: {state.url if state else 'N/A'}")
            await self._raise_if_stopped_or_paused()

            history_summary_str = self._summarize_browsing_history(max_steps=5, max_chars=1500)

            self.message_manager.add_state_message(
                state=state,
                actions=self.state.last_action, # type: ignore[call-arg]
                result=self.state.last_result,
                step_info=current_custom_step_info, # Use the created CustomAgentStepInfo
                use_vision=self.settings.use_vision,
                history_summary=history_summary_str # type: ignore[call-arg]
            )

            if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
                await self._run_planner()
            input_messages = self.message_manager.get_messages()
            tokens = self._message_manager.state.history.current_tokens

            try:
                model_output = await self.get_next_action(input_messages)
                self._log_response(model_output)

                # self.state.n_steps is incremented here, AFTER CustomAgentStepInfo was created with the *current* step number
                # This is fine, as the prompt needs the current step, and n_steps tracks completed/next step.

                if self.register_new_step_callback:
                    await self.register_new_step_callback(state, model_output, self.state.n_steps +1) # n_steps will be for the *next* step

                if self.settings.save_conversation_path:
                    target = self.settings.save_conversation_path + f'_{self.state.n_steps +1}.txt'
                    save_conversation(input_messages, model_output, target,
                                      self.settings.save_conversation_path_encoding)

                if self.model_name != "deepseek-reasoner":
                    self.message_manager._remove_state_message_by_index(-1) # type: ignore[attr-defined]
                await self._raise_if_stopped_or_paused()
            except Exception as e:
                self.message_manager._remove_state_message_by_index(-1) # type: ignore[attr-defined]
                raise e

            result = await self.multi_act(model_output.action) # type: ignore
            
            # Update step_info's memory (which is current_custom_step_info) with model output
            self.update_step_info(model_output, current_custom_step_info) # type: ignore
            # Persist the updated memory for the next step
            self.current_task_memory = current_custom_step_info.memory
            
            # Increment n_steps after all actions for the current step are done and memory is updated.
            self.state.n_steps += 1


            for ret_ in result:
                if ret_.extracted_content and "Extracted page" in ret_.extracted_content:
                    if ret_.extracted_content[:100] not in self.state.extracted_content:
                        self.state.extracted_content += ret_.extracted_content
            self.state.last_result = result
            self.state.last_action = model_output.action
            if len(result) > 0 and result[-1].is_done:
                if not self.state.extracted_content:
                    # If step_info's memory was used for CustomAgentStepInfo it might be outdated here.
                    # Use current_task_memory which should be the most up-to-date.
                    self.state.extracted_content = self.current_task_memory 
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
            logger.debug("Entering CustomAgent.step finally block.") # DEBUG
            step_end_time = time.time()
            actions_telemetry = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output and hasattr(model_output, 'action') and model_output.action else []
            
            logger.debug("Attempting to capture telemetry.") # DEBUG
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps, # Note: n_steps was already incremented
                    actions=actions_telemetry,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=[r.error for r in result if r.error] if result else ['No result after step execution'], # Modified for clarity
                )
            )
            logger.debug("Telemetry captured.") # DEBUG

            if not result:
                logger.debug("No result from multi_act, returning from step.") # DEBUG
                return

            if state and model_output:
                logger.debug(f"Calling _make_history_item with model_output: {type(model_output)}, state: {type(state)}, result: {type(result)}") # DEBUG
                metadata = StepMetadata(
                    step_number=self.state.n_steps, # n_steps was already incremented
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)
                logger.debug("_make_history_item finished.") # DEBUG
            else:
                logger.debug("Skipping _make_history_item due to no state or model_output.") # DEBUG
            
            # Log final state before returning from step
            logger.debug(f"CustomAgent.step state before return: n_steps={self.state.n_steps}, stopped={self.state.stopped}, paused={self.state.paused}, consecutive_failures={self.state.consecutive_failures}, last_result_count={len(self.state.last_result) if self.state.last_result else 0}")
            if self.state.last_result:
                for i, res_item in enumerate(self.state.last_result):
                    logger.debug(f"  last_result[{i}]: error='{res_item.error}', is_done={res_item.is_done}")

            logger.debug("Exiting CustomAgent.step finally block.") # DEBUG

    # New: modified to accept ReplayTaskDetails at replay mode
    async def run(self, task_input: Union[str, ReplayTaskDetails], max_steps: int = 100) -> Optional[AgentHistoryList]:
        """
        Run the agent to complete the task.
        If task_input is ReplayTaskDetails, it runs in replay mode.
        Otherwise, it runs in autonomous mode.
        """
        self.state.start_time = time.time()
        self.state.task_input = task_input
        self.state.max_steps = max_steps

        if isinstance(task_input, ReplayTaskDetails) and task_input.mode == "replay":
            logger.info(f"🚀 Starting agent in REPLAY mode for trace: {task_input.trace_path}")
            if not self.browser_context:
                logger.error("Replay mode: Browser context is not available.")
                return None
            
            # Ensure there is a page to replay on
            if not self.page or self.page.is_closed():
                logger.info("Replay mode: self.page is not valid. Attempting to get/create a page.")
                playwright_context = getattr(self.browser_context, "playwright_context", None)
                if playwright_context and playwright_context.pages:
                    self.page = playwright_context.pages[0]
                    await self.page.bring_to_front()
                    logger.info(f"Replay mode: Using existing page: {self.page.url}")
                elif playwright_context:
                    self.page = await playwright_context.new_page()
                    logger.info(f"Replay mode: Created new page: {self.page.url}")
                else:
                    logger.error("Replay mode: playwright_context is None, cannot create or get a page.")
                    return None
            
            try:
                trace_events = load_trace(task_input.trace_path)
                if not trace_events:
                    logger.warning(f"Replay mode: No events found in trace file: {task_input.trace_path}")
                    return None
                
                replayer = TraceReplayer(self.page, trace_events)
                logger.info(f"Replayer initialized. Starting playback at speed: {task_input.speed}x")
                await replayer.play(speed=task_input.speed)
                logger.info(f"🏁 Replay finished for trace: {task_input.trace_path}")
            except Drift as d:
                drift_message = getattr(d, "message", str(d))
                logger.error(f"💣 DRIFT DETECTED during replay of {task_input.trace_path}: {drift_message}")
                if d.event:
                    logger.error(f"   Drift occurred at event: {json.dumps(d.event)}")
            except FileNotFoundError:
                logger.error(f"Replay mode: Trace file not found at {task_input.trace_path}")
            except Exception as e:
                logger.exception(f"Replay mode: An unexpected error occurred during replay of {task_input.trace_path}")
            finally:
                # Decide if browser/context should be closed after replay based on agent settings (e.g., keep_browser_open)
                # For now, let's assume it follows the general agent cleanup logic if applicable, or stays open.
                pass
            return None # Replay mode doesn't return standard agent history

        # Autonomous mode logic continues below
        elif isinstance(task_input, str):
            if task_input != self.task:
                logger.info(f"Autonomous run: Task updated from '{self.task}' to '{task_input}'")
                self.task = task_input
                self._message_manager.task = self.task # Update message manager's task
                 # Reset or update initial messages in message manager if task significantly changes
                if hasattr(self._message_manager.state.history, "history") and isinstance(self._message_manager.state.history.history, list): # type: ignore[attr-defined]
                    self._message_manager.state.history.history.clear() # type: ignore[attr-defined]
                
                if hasattr(self._message_manager, "add_initial_messages"):
                    self._message_manager.add_initial_messages(self.task, self.add_infos) # type: ignore
                else:
                    logger.warning("CustomMessageManager does not have add_initial_messages method.")
            elif not isinstance(task_input, str):
                 logger.warning(f"Autonomous run: task_input is not a string ({type(task_input)}). Using existing task: {self.task}")


            logger.info(f"Starting autonomous agent run for task: '{self.task}', max_steps: {max_steps}")
            logger.debug(f"CustomAgent: About to call super().run(max_steps={max_steps})") # DEBUG
            # Use the base Agent.run() method for the main loop and its own try/finally for telemetry etc.
            history: Optional[AgentHistoryList] = await super().run(max_steps=max_steps)
            logger.debug(f"CustomAgent: super().run() returned. History is None: {history is None}") # DEBUG
            if history and hasattr(history, 'history'):
                logger.debug(f"CustomAgent: History length: {len(history.history) if history.history else 0}") # DEBUG

            # After autonomous run, UserInputTracker history persistence is handled by the UI's explicit stop recording.
            # The agent itself, when run with a string task, should not be responsible for this.
            # Removing the block that attempted to save UserInputTracker traces here.
            
            return history

    def _convert_input_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        converted_messages = []
        for msg in messages:
            msg_item = {}
            if isinstance(msg, HumanMessage):
                msg_item["role"] = "user"
                msg_item["content"] = msg.content
            elif isinstance(msg, AIMessage):
                msg_item["role"] = "assistant"
                # Handle tool calls if present
                if msg.tool_calls:
                    msg_item["content"] = None # Standard AIMessage content is None if tool_calls are present
                    msg_item["tool_calls"] = msg.tool_calls
                else:
                    msg_item["content"] = msg.content
            elif hasattr(msg, 'role') and hasattr(msg, 'content'): # For generic BaseMessage with role and content
                 msg_item["role"] = getattr(msg, "role", "unknown")
                 msg_item["content"] = getattr(msg, "content", "")
            else:
                # Fallback or skip if message type is not directly convertible
                logger.warning(f"Skipping message of unhandled type: {type(msg)}")
                continue

            # Add reasoning_content for tool_code type messages if available
            if msg_item.get("type") == "tool_code" and isinstance(msg, AIMessage) and hasattr(msg, 'reasoning_content'):
                reasoning_content = getattr(msg, "reasoning_content", None)
                if reasoning_content:
                    msg_item["reasoning_content"] = reasoning_content
            converted_messages.append(msg_item)
        return converted_messages

    def _parse_model_output(self, output: str, ActionModel: Type[BaseModel]) -> CustomAgentOutput:
        try:
            if not hasattr(self, 'AgentOutput') or not self.AgentOutput:
                self._setup_action_models() # Sets self.AgentOutput

            extracted_output: Union[str, Dict[Any, Any]] = extract_json_from_model_output(output)
            parsed_data: CustomAgentOutput

            if isinstance(extracted_output, dict):
                # If it's already a dict, assume it's valid JSON and Pydantic can handle it
                parsed_data = self.AgentOutput.model_validate(extracted_output)
            elif isinstance(extracted_output, str):
                # If it's a string, try to repair it then parse
                repaired_json_string = repair_json(extracted_output, return_objects=False)
                if not isinstance(repaired_json_string, str):
                    logger.error(f"repair_json with return_objects=False did not return a string. Got: {type(repaired_json_string)}. Falling back to original extracted string.")
                    # Fallback or raise error. Forcing to string for now.
                    repaired_json_string = str(extracted_output) # Fallback to the original extracted string if repair fails badly
                parsed_data = self.AgentOutput.model_validate_json(repaired_json_string)
            else:
                raise ValueError(f"Unexpected output type from extract_json_from_model_output: {type(extracted_output)}")
            
            # Ensure the final parsed_data is indeed CustomAgentOutput
            if not isinstance(parsed_data, CustomAgentOutput):
                logger.warning(f"Parsed data is type {type(parsed_data)}, not CustomAgentOutput. Attempting conversion or default.")
                # This might happen if self.AgentOutput.model_validate/model_validate_json doesn't return the precise
                # CustomAgentOutput type but a compatible one (e.g. base AgentOutput).
                # We need to ensure it has the CustomAgentBrain structure.
                action_list = getattr(parsed_data, 'action', [])
                current_state_data = getattr(parsed_data, 'current_state', None)

                if isinstance(current_state_data, CustomAgentBrain):
                    parsed_data = self.AgentOutput(action=action_list, current_state=current_state_data)
                elif isinstance(current_state_data, dict):
                    try:
                        brain = CustomAgentBrain(**current_state_data)
                        parsed_data = self.AgentOutput(action=action_list, current_state=brain)
                    except Exception as brain_ex:
                        logger.error(f"Could not construct CustomAgentBrain from dict: {brain_ex}. Falling back to error brain.")
                        error_brain = CustomAgentBrain(
                            evaluation_previous_goal="Error",
                            important_contents="Failed to reconstruct agent brain during parsing.",
                            thought="Critical error in parsing agent state.",
                            next_goal="Retry or report error."
                        )
                        parsed_data = self.AgentOutput(action=action_list, current_state=error_brain)
                else:
                    logger.error("current_state is missing or not CustomAgentBrain/dict. Falling back to error brain.")
                    error_brain = CustomAgentBrain(
                        evaluation_previous_goal="Error",
                        important_contents="Missing or invalid agent brain during parsing.",
                        thought="Critical error in parsing agent state.",
                        next_goal="Retry or report error."
                    )
                    # Ensure action_list is compatible if it came from a different model type
                    # For simplicity, if we have to create an error brain, we might also want to clear actions
                    # or ensure they are valid ActionModel instances. For now, passing them as is.
                    parsed_data = self.AgentOutput(action=action_list, current_state=error_brain)
            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing model output: {e}\\nRaw output:\\n{output}", exc_info=True)
            if not hasattr(self, 'AgentOutput') or not self.AgentOutput:
                self._setup_action_models() # Ensure self.AgentOutput is set up for fallback
            
            error_brain = CustomAgentBrain(
                evaluation_previous_goal="Error",
                important_contents=f"Parsing error: {str(e)}",
                thought=f"Failed to parse LLM output. Error: {str(e)}",
                next_goal="Retry or report error."
            )
            return self.AgentOutput(action=[], current_state=error_brain)

        # pass # Original empty implementation