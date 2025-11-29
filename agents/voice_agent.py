"""
Base Voice Agent class for all voice-enabled agents.
Combines LangGraph orchestration with Cartesia voice and multi-LLM support.

NO OpenAI - Uses Claude, Gemini, OpenRouter for LLM and Cartesia for voice.
"""

import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from enum import Enum
from datetime import datetime

from langgraph.graph import StateGraph, END

from core.base_graph import (
    BaseAgent,
    AgentState,
    AgentMode,
    MultiModalMixin,
    LLMIntegrationMixin,
    ErrorHandlingMixin,
)
from llm import LLMConfig, LLMProvider
from llm.router import TaskType
from voice import (
    CartesiaConfig,
    LiveKitConfig,
    LiveKitCartesiaClient,
    create_voice_session,
)

logger = logging.getLogger(__name__)


class VoiceAgentState(Enum):
    """Voice agent conversation states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    voice_used: bool = False


@dataclass
class VoiceAgentConfig:
    """Configuration for voice agents"""
    # Agent identity
    agent_name: str = "Voice Assistant"
    agent_description: str = "A helpful voice-enabled AI assistant"

    # System prompt
    system_prompt: Optional[str] = None

    # LLM settings
    llm_config: Optional[LLMConfig] = None
    default_provider: Optional[LLMProvider] = None
    task_type: TaskType = TaskType.GENERAL

    # Voice settings
    cartesia_config: Optional[CartesiaConfig] = None
    voice_id: Optional[str] = None

    # LiveKit settings (optional - for WebRTC sessions)
    livekit_config: Optional[LiveKitConfig] = None
    enable_livekit: bool = False

    # Conversation settings
    max_history_turns: int = 20
    greeting_message: Optional[str] = None


class VoiceAgent(
    MultiModalMixin,
    LLMIntegrationMixin,
    ErrorHandlingMixin,
    BaseAgent
):
    """
    Base class for voice-enabled agents.

    Provides:
    - Multi-LLM support (Claude, Gemini, OpenRouter)
    - Cartesia TTS/STT
    - Optional LiveKit WebRTC integration
    - Conversation history management
    - LangGraph workflow orchestration

    NO OpenAI dependencies.
    """

    def __init__(
        self,
        config: Optional[VoiceAgentConfig] = None,
        mode: AgentMode = AgentMode.VOICE,
    ):
        self.config = config or VoiceAgentConfig()

        # Initialize base agent
        super().__init__(
            agent_type=self.config.agent_name,
            mode=mode,
        )

        # Initialize LLM
        self.initialize_llm(
            config=self.config.llm_config,
            default_task_type=self.config.task_type,
        )

        # Voice state
        self._voice_state = VoiceAgentState.IDLE
        self._livekit_client: Optional[LiveKitCartesiaClient] = None

        # Conversation management
        self._conversation_history: List[ConversationTurn] = []

        # Event handlers
        self.on_state_change: Optional[Callable[[VoiceAgentState], Any]] = None
        self.on_response_ready: Optional[Callable[[str], Any]] = None

        logger.info(f"Voice agent initialized: {self.config.agent_name}")

    @property
    def voice_state(self) -> VoiceAgentState:
        """Current voice processing state"""
        return self._voice_state

    @property
    def conversation_history(self) -> List[ConversationTurn]:
        """Get conversation history"""
        return self._conversation_history.copy()

    def _set_voice_state(self, state: VoiceAgentState):
        """Update voice state and notify listeners"""
        old_state = self._voice_state
        self._voice_state = state

        if self.on_state_change and old_state != state:
            if asyncio.iscoroutinefunction(self.on_state_change):
                asyncio.create_task(self.on_state_change(state))
            else:
                self.on_state_change(state)

        logger.debug(f"Voice state: {old_state.value} -> {state.value}")

    async def start_voice_session(
        self,
        room_name: Optional[str] = None,
        participant_name: Optional[str] = None,
    ) -> bool:
        """
        Start a voice session with LiveKit and Cartesia.

        Args:
            room_name: Optional LiveKit room name
            participant_name: Optional participant name

        Returns:
            True if session started successfully
        """
        try:
            # Enable voice mode (Cartesia only, no LiveKit)
            await self.enable_voice_mode(
                config=self.config.cartesia_config,
                voice_id=self.config.voice_id,
            )

            # If LiveKit is enabled, also connect to room
            if self.config.enable_livekit:
                livekit_config = self.config.livekit_config or LiveKitConfig()
                if room_name:
                    livekit_config.room_name = room_name
                if participant_name:
                    livekit_config.participant_name = participant_name

                self._livekit_client = LiveKitCartesiaClient(livekit_config)
                self._livekit_client.on_transcription = self._handle_transcription

                if not await self._livekit_client.connect():
                    logger.error("Failed to connect to LiveKit")
                    return False

            self._set_voice_state(VoiceAgentState.IDLE)

            # Send greeting if configured
            if self.config.greeting_message:
                await self.speak(self.config.greeting_message)

            logger.info("Voice session started")
            return True

        except Exception as e:
            logger.error(f"Failed to start voice session: {e}")
            self._set_voice_state(VoiceAgentState.ERROR)
            return False

    async def stop_voice_session(self):
        """Stop the current voice session"""
        if self._livekit_client:
            await self._livekit_client.disconnect()
            self._livekit_client = None

        await self.disable_voice_mode()
        self._set_voice_state(VoiceAgentState.IDLE)
        logger.info("Voice session stopped")

    async def _handle_transcription(self, text: str):
        """Handle incoming transcription from LiveKit/Cartesia"""
        if not text.strip():
            return

        logger.info(f"Received transcription: {text}")

        # Process through the agent
        response = await self.process_input(text)

        if response and self.on_response_ready:
            if asyncio.iscoroutinefunction(self.on_response_ready):
                await self.on_response_ready(response)
            else:
                self.on_response_ready(response)

    async def speak(self, text: str) -> bool:
        """
        Synthesize and speak text.

        Args:
            text: Text to speak

        Returns:
            True if successful
        """
        if not text.strip():
            return False

        self._set_voice_state(VoiceAgentState.SPEAKING)

        try:
            # If LiveKit is connected, speak through room
            if self._livekit_client and self._livekit_client.is_connected():
                result = await self._livekit_client.speak(text)
            elif self.voice_enabled:
                # Otherwise, just generate audio (caller handles playback)
                await self.generate_voice_output(text)
                result = True
            else:
                logger.warning("Voice not enabled, cannot speak")
                result = False

            return result

        except Exception as e:
            logger.error(f"Speech error: {e}")
            return False

        finally:
            self._set_voice_state(VoiceAgentState.IDLE)

    async def listen(self, audio_data: bytes) -> str:
        """
        Transcribe audio input.

        Args:
            audio_data: PCM s16le audio bytes

        Returns:
            Transcribed text
        """
        self._set_voice_state(VoiceAgentState.LISTENING)

        try:
            if self._livekit_client and self._livekit_client.is_connected():
                return await self._livekit_client.transcribe(audio_data)
            elif self.voice_enabled:
                return await self.process_voice_input(audio_data)
            else:
                logger.warning("Voice not enabled, cannot listen")
                return ""

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

        finally:
            self._set_voice_state(VoiceAgentState.IDLE)

    async def process_input(self, input_data: str) -> str:
        """
        Process user input and generate response.

        Args:
            input_data: User's text input

        Returns:
            Agent's response
        """
        self._set_voice_state(VoiceAgentState.PROCESSING)

        try:
            # Add to conversation history
            self._add_to_history("user", input_data)

            # Build conversation context
            history = self._build_history_for_llm()

            # Get system prompt
            system = self._get_system_prompt()

            # Generate response
            response = await self.generate_response(
                user_message=input_data,
                system=system,
                conversation_history=history,
                provider=self.config.default_provider,
            )

            # Add response to history
            self._add_to_history("assistant", response)

            # Speak the response if in voice mode
            if self.mode == AgentMode.VOICE:
                await self.speak(response)

            return response

        except Exception as e:
            logger.error(f"Processing error: {e}")
            error_response = "I apologize, but I encountered an error. Please try again."
            self._add_to_history("assistant", error_response)
            return error_response

        finally:
            self._set_voice_state(VoiceAgentState.IDLE)

    async def stream_response(
        self,
        input_data: str,
        speak_chunks: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Stream response generation with optional speech synthesis.

        Args:
            input_data: User's text input
            speak_chunks: Whether to speak response chunks

        Yields:
            Response tokens
        """
        self._set_voice_state(VoiceAgentState.PROCESSING)

        try:
            self._add_to_history("user", input_data)
            history = self._build_history_for_llm()
            system = self._get_system_prompt()

            full_response = []

            # Stream from LLM
            async for token in super().stream_response(
                user_message=input_data,
                system=system,
                conversation_history=history,
                provider=self.config.default_provider,
            ):
                full_response.append(token)
                yield token

            # Add complete response to history
            complete_response = ''.join(full_response)
            self._add_to_history("assistant", complete_response)

            # Speak the complete response if requested
            if speak_chunks and self.mode == AgentMode.VOICE:
                await self.speak(complete_response)

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield "I apologize, but I encountered an error."

        finally:
            self._set_voice_state(VoiceAgentState.IDLE)

    def _add_to_history(self, role: str, content: str):
        """Add a turn to conversation history"""
        turn = ConversationTurn(
            role=role,
            content=content,
            voice_used=self.mode == AgentMode.VOICE
        )
        self._conversation_history.append(turn)

        # Trim if exceeds max
        if len(self._conversation_history) > self.config.max_history_turns:
            self._conversation_history = self._conversation_history[-self.config.max_history_turns:]

    def _build_history_for_llm(self) -> List[Dict[str, str]]:
        """Build history in format expected by LLM"""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self._conversation_history[:-1]  # Exclude current message
        ]

    def _get_system_prompt(self) -> str:
        """Get system prompt for this agent"""
        if self.config.system_prompt:
            return self.config.system_prompt

        return f"""You are {self.config.agent_name}, {self.config.agent_description}.

You are a voice-enabled AI assistant. Keep your responses conversational and concise.
When speaking, use natural language and avoid complex formatting that doesn't work in speech.

Current date: {datetime.now().strftime('%Y-%m-%d')}
"""

    def clear_history(self):
        """Clear conversation history"""
        self._conversation_history = []
        logger.info("Conversation history cleared")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph for this agent"""
        # Simple graph for basic agents
        # Subclasses can override for more complex workflows
        graph = StateGraph(AgentState)

        # Single processing node
        graph.add_node("process", self._process_node)

        # Entry point
        graph.set_entry_point("process")

        # End after processing
        graph.add_edge("process", END)

        return graph

    async def _process_node(self, state: dict) -> dict:
        """Default processing node"""
        current_task = state.get("current_task", "")

        if current_task:
            response = await self.process_input(current_task)
            state["messages"] = state.get("messages", []) + [
                {"role": "assistant", "content": response}
            ]

        return state

    async def close(self):
        """Clean up resources"""
        await self.stop_voice_session()
        await self.close_llm()
        logger.info(f"Voice agent closed: {self.config.agent_name}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
