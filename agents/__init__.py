"""
Voice-enabled agents for LangGraph framework.

Three specialized agents:
- GeneralAssistant: Conversational AI for everyday tasks
- CodeAssistant: Coding helper and reviewer
- TaskManager: Productivity and task management

All agents support voice interaction via Cartesia.
NO OpenAI dependencies.
"""

from agents.voice_agent import (
    VoiceAgent,
    VoiceAgentConfig,
    VoiceAgentState,
    ConversationTurn,
)
from agents.general_assistant import (
    GeneralAssistant,
    create_general_assistant,
)
from agents.code_assistant import (
    CodeAssistant,
    CodeLanguage,
    CodeContext,
    create_code_assistant,
)
from agents.task_manager import (
    TaskManager,
    Task,
    Project,
    TaskPriority,
    TaskStatus,
    create_task_manager,
)

__all__ = [
    # Base Voice Agent
    "VoiceAgent",
    "VoiceAgentConfig",
    "VoiceAgentState",
    "ConversationTurn",
    # General Assistant
    "GeneralAssistant",
    "create_general_assistant",
    # Code Assistant
    "CodeAssistant",
    "CodeLanguage",
    "CodeContext",
    "create_code_assistant",
    # Task Manager
    "TaskManager",
    "Task",
    "Project",
    "TaskPriority",
    "TaskStatus",
    "create_task_manager",
]
