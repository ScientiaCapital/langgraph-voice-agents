"""
Base LangGraph implementation for multi-modal agent framework.
Provides foundation for all agents with voice and text capabilities.

Voice: Cartesia TTS/STT (NO OpenAI)
LLM: Claude, Gemini, OpenRouter (NO OpenAI)
Transport: LiveKit WebRTC
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import uuid
import asyncio
import logging
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver

from llm import LLMRouter, LLMProvider, LLMConfig
from llm.router import TaskType
from voice import CartesiaClient, CartesiaConfig

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Agent execution modes"""
    TEXT = "text"
    VOICE = "voice"
    HYBRID = "hybrid"


@dataclass
class AgentState:
    """Shared state for all agents in the framework"""

    # Core identification
    session_id: str
    agent_type: str
    mode: AgentMode

    # Task management
    current_task: Optional[str] = None
    task_history: List[str] = None
    task_context: Dict[str, Any] = None

    # Communication
    messages: List[Dict[str, Any]] = None
    voice_input: Optional[str] = None
    voice_output: Optional[str] = None

    # MCP tool integration
    sequential_thinking_result: Optional[Dict] = None
    serena_analysis: Optional[Dict] = None
    context7_docs: Optional[Dict] = None

    # LiveKit session info
    livekit_room_id: Optional[str] = None
    livekit_participant_id: Optional[str] = None

    # Error handling
    errors: List[str] = None
    retry_count: int = 0

    def __post_init__(self):
        """Initialize mutable fields"""
        if self.task_history is None:
            self.task_history = []
        if self.task_context is None:
            self.task_context = {}
        if self.messages is None:
            self.messages = []
        if self.errors is None:
            self.errors = []


class BaseAgent(ABC):
    """Abstract base class for all LangGraph agents"""

    def __init__(
        self,
        agent_type: str,
        mode: AgentMode = AgentMode.TEXT,
        checkpointer_path: Optional[str] = None
    ):
        self.agent_type = agent_type
        self.mode = mode
        self.session_id = str(uuid.uuid4())

        # Initialize checkpointer for state persistence
        self.checkpointer = SqliteSaver.from_conn_string(
            checkpointer_path or f":memory:{self.session_id}"
        )

        # Initialize the LangGraph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

        logger.info(f"Initialized {agent_type} agent in {mode.value} mode")

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for this agent"""
        pass

    @abstractmethod
    async def process_input(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process input and return response"""
        pass

    def create_initial_state(
        self,
        task: str,
        mode: Optional[AgentMode] = None
    ) -> AgentState:
        """Create initial state for a new task"""
        return AgentState(
            session_id=self.session_id,
            agent_type=self.agent_type,
            mode=mode or self.mode,
            current_task=task,
            task_context={"created_at": datetime.now().isoformat()}
        )

    async def execute_workflow(
        self,
        initial_state: AgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the agent workflow with given state"""
        try:
            config = config or {"configurable": {"thread_id": self.session_id}}

            result = await self.app.ainvoke(
                initial_state.__dict__,
                config=config
            )

            logger.info(f"Workflow completed for {self.agent_type}")
            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    async def get_current_state(self, thread_id: Optional[str] = None) -> Optional[Dict]:
        """Get current state from checkpointer"""
        thread_id = thread_id or self.session_id
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = await self.app.aget_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None

    async def update_state(
        self,
        updates: Dict[str, Any],
        thread_id: Optional[str] = None
    ) -> bool:
        """Update current state"""
        thread_id = thread_id or self.session_id
        config = {"configurable": {"thread_id": thread_id}}

        try:
            await self.app.aupdate_state(config, updates)
            logger.debug(f"State updated for {self.agent_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
            return False


class MultiModalMixin:
    """
    Mixin for agents that support both text and voice modes.

    Uses Cartesia for TTS/STT (NO OpenAI).
    LiveKit is handled separately for WebRTC transport.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._voice_enabled = False
        self._cartesia_client: Optional[CartesiaClient] = None
        self._voice_config: Optional[CartesiaConfig] = None

    async def enable_voice_mode(
        self,
        config: Optional[CartesiaConfig] = None,
        voice_id: Optional[str] = None
    ):
        """
        Enable voice capabilities using Cartesia.

        Args:
            config: Optional Cartesia configuration
            voice_id: Optional voice ID override for TTS
        """
        try:
            self._voice_config = config or CartesiaConfig()
            if voice_id:
                self._voice_config.voice_id = voice_id

            self._cartesia_client = CartesiaClient(self._voice_config)
            await self._cartesia_client.connect()
            self._voice_enabled = True

            logger.info(f"Voice mode enabled for {self.agent_type} (Cartesia)")

        except Exception as e:
            logger.error(f"Failed to enable voice mode: {e}")
            self._voice_enabled = False
            raise

    async def disable_voice_mode(self):
        """Disable voice capabilities"""
        if self._cartesia_client:
            await self._cartesia_client.disconnect()
            self._cartesia_client = None
        self._voice_enabled = False
        self._voice_config = None

        logger.info(f"Voice mode disabled for {self.agent_type}")

    @property
    def voice_enabled(self) -> bool:
        """Check if voice mode is enabled"""
        return self._voice_enabled

    @property
    def cartesia_client(self) -> Optional[CartesiaClient]:
        """Get the Cartesia client if voice is enabled"""
        return self._cartesia_client

    async def process_voice_input(self, audio_data: bytes) -> str:
        """
        Process voice input and return transcribed text.
        Uses Cartesia STT (ink-whisper model).

        Args:
            audio_data: PCM s16le audio bytes

        Returns:
            Transcribed text
        """
        if not self._voice_enabled or not self._cartesia_client:
            raise RuntimeError("Voice mode not enabled")

        return await self._cartesia_client.listen(audio_data)

    async def generate_voice_output(self, text: str) -> bytes:
        """
        Generate voice output from text.
        Uses Cartesia TTS (sonic-2 model).

        Args:
            text: Text to synthesize

        Returns:
            Audio bytes (PCM s16le)
        """
        if not self._voice_enabled or not self._cartesia_client:
            raise RuntimeError("Voice mode not enabled")

        # Collect all audio chunks
        audio_chunks = []
        async for chunk in self._cartesia_client.speak(text):
            audio_chunks.append(chunk)

        return b''.join(audio_chunks)

    async def stream_voice_output(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream voice output for low-latency playback.
        Yields audio chunks as they're synthesized.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks (PCM s16le)
        """
        if not self._voice_enabled or not self._cartesia_client:
            raise RuntimeError("Voice mode not enabled")

        async for chunk in self._cartesia_client.speak(text):
            yield chunk


class ErrorHandlingMixin:
    """Mixin for standardized error handling across agents"""

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    async def execute_with_retry(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ):
        """Execute function with retry logic"""
        max_retries = max_retries or self.MAX_RETRIES

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                    raise

                delay = self.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} in {delay}s")
                await asyncio.sleep(delay)

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Standardized error handling"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "agent_type": getattr(self, 'agent_type', 'unknown')
        }

        logger.error(f"Agent error: {error_info}")
        return error_info


class LLMIntegrationMixin:
    """
    Mixin for agents that need LLM capabilities.

    Provides access to multi-LLM router with automatic failover.
    Supports Claude, Gemini, and OpenRouter (NO OpenAI).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm_router: Optional[LLMRouter] = None
        self._llm_config: Optional[LLMConfig] = None
        self._default_task_type: TaskType = TaskType.GENERAL

    def initialize_llm(
        self,
        config: Optional[LLMConfig] = None,
        default_task_type: TaskType = TaskType.GENERAL
    ):
        """
        Initialize the LLM router for this agent.

        Args:
            config: Optional LLM configuration
            default_task_type: Default task type for provider selection
        """
        self._llm_config = config or LLMConfig()
        self._llm_router = LLMRouter(self._llm_config)
        self._default_task_type = default_task_type

        available = self._llm_router.available_providers
        logger.info(f"LLM initialized for {getattr(self, 'agent_type', 'unknown')}. "
                   f"Available providers: {[p.value for p in available]}")

    @property
    def llm(self) -> Optional[LLMRouter]:
        """Get the LLM router"""
        return self._llm_router

    @property
    def llm_available(self) -> bool:
        """Check if LLM is initialized and has available providers"""
        return self._llm_router is not None and len(self._llm_router.available_providers) > 0

    async def generate_response(
        self,
        user_message: str,
        system: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        task_type: Optional[TaskType] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using the LLM router.

        Args:
            user_message: User's message
            system: System prompt
            conversation_history: Previous messages
            task_type: Task type for provider selection
            provider: Specific provider to use
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        if not self._llm_router:
            raise RuntimeError("LLM not initialized. Call initialize_llm() first.")

        from llm.provider import Message

        # Convert history to Message objects
        history = None
        if conversation_history:
            history = [
                Message(role=m["role"], content=m["content"])
                for m in conversation_history
            ]

        return await self._llm_router.chat(
            user_message=user_message,
            conversation_history=history,
            system=system,
            provider=provider,
            task_type=task_type or self._default_task_type,
            **kwargs
        )

    async def stream_response(
        self,
        user_message: str,
        system: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        task_type: Optional[TaskType] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response for real-time display.

        Args:
            user_message: User's message
            system: System prompt
            conversation_history: Previous messages
            task_type: Task type for provider selection
            provider: Specific provider to use
            **kwargs: Additional parameters

        Yields:
            Response tokens as they're generated
        """
        if not self._llm_router:
            raise RuntimeError("LLM not initialized. Call initialize_llm() first.")

        from llm.provider import Message

        history = None
        if conversation_history:
            history = [
                Message(role=m["role"], content=m["content"])
                for m in conversation_history
            ]

        async for token in self._llm_router.stream_chat(
            user_message=user_message,
            conversation_history=history,
            system=system,
            provider=provider,
            task_type=task_type or self._default_task_type,
            **kwargs
        ):
            yield token

    async def close_llm(self):
        """Close LLM connections"""
        if self._llm_router:
            await self._llm_router.close()
            self._llm_router = None
            logger.info(f"LLM closed for {getattr(self, 'agent_type', 'unknown')}")


# Utility functions for graph building

def create_conditional_edge(condition_func, mapping: Dict[str, str]):
    """Helper to create conditional edges in LangGraph"""
    def conditional(state):
        result = condition_func(state)
        return mapping.get(result, END)

    return conditional


def create_tool_node(tools: List, tool_executor: Optional[ToolExecutor] = None):
    """Helper to create tool execution nodes"""
    if tool_executor is None:
        tool_executor = ToolExecutor(tools)

    async def tool_node(state):
        """Execute tools based on last message"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if last_message and hasattr(last_message, "tool_calls"):
            return {"messages": [await tool_executor.ainvoke(last_message)]}

        return {"messages": []}

    return tool_node