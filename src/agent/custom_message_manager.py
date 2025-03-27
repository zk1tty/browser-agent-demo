from __future__ import annotations

import logging
import pdb
from typing import List, Optional, Type, Dict

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, AgentStepInfo, ActionModel
from browser_use.browser.views import BrowserState
from browser_use.agent.message_manager.service import MessageManagerSettings
from browser_use.agent.views import ActionResult, AgentOutput, AgentStepInfo, MessageManagerState
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI
from ..utils.llm import DeepSeekR1ChatOpenAI
from .custom_prompts import CustomAgentMessagePrompt

logger = logging.getLogger(__name__)


class CustomMessageManagerSettings(MessageManagerSettings):
    agent_prompt_class: Type[AgentMessagePrompt] = AgentMessagePrompt


class CustomMessageManager(MessageManager):
    def __init__(
            self,
            task: str,
            system_message: SystemMessage,
            settings: MessageManagerSettings = MessageManagerSettings(),
            state: MessageManagerState = MessageManagerState(),
    ):
        super().__init__(
            task=task,
            system_message=system_message,
            settings=settings,
            state=state
        )

    def _init_messages(self) -> None:
        """Initialize the message history with system message, context, task, and other initial messages"""
        self._add_message_with_tokens(self.system_prompt)
        self.context_content = ""

        if self.settings.message_context:
            self.context_content += 'Context for the task' + self.settings.message_context

        if self.settings.sensitive_data:
            info = f'Here are placeholders for sensitive data: {list(self.settings.sensitive_data.keys())}'
            info += 'To use them, write <secret>the placeholder name</secret>'
            self.context_content += info

        if self.settings.available_file_paths:
            filepaths_msg = f'Here are file paths you can use: {self.settings.available_file_paths}'
            self.context_content += filepaths_msg

        if self.context_content:
            context_message = HumanMessage(content=self.context_content)
            self._add_message_with_tokens(context_message)

    def cut_messages(self):
        """Get current message list, potentially trimmed to max tokens"""
        diff = self.state.history.current_tokens - self.settings.max_input_tokens
        min_message_len = 2 if self.context_content is not None else 1

        while diff > 0 and len(self.state.history.messages) > min_message_len:
            self.state.history.remove_message(min_message_len)  # always remove the oldest message
            diff = self.state.history.current_tokens - self.settings.max_input_tokens

    def add_state_message(
            self,
            state: BrowserState,
            actions: Optional[List[ActionModel]] = None,
            result: Optional[List[ActionResult]] = None,
            step_info: Optional[AgentStepInfo] = None,
            use_vision=True,
    ) -> None:
        """Add browser state as human message"""
        # otherwise add state message and result to next message (which will not stay in memory)
        state_message = self.settings.agent_prompt_class(
            state,
            actions,
            result,
            include_attributes=self.settings.include_attributes,
            step_info=step_info,
        ).get_user_message(use_vision)
        self._add_message_with_tokens(state_message)

    def _remove_state_message_by_index(self, remove_ind=-1) -> None:
        """Remove state message by index from history"""
        i = len(self.state.history.messages) - 1
        remove_cnt = 0
        while i >= 0:
            if isinstance(self.state.history.messages[i].message, HumanMessage):
                remove_cnt += 1
            if remove_cnt == abs(remove_ind):
                self.state.history.messages.pop(i)
                break
            i -= 1
