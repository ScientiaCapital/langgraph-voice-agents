"""
General Assistant - Conversational voice agent for everyday tasks.

Handles:
- General conversation and Q&A
- Information lookup and summarization
- Creative writing and brainstorming
- Explanations and tutoring

Uses Claude or Gemini for natural conversation.
NO OpenAI dependencies.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END

from agents.voice_agent import VoiceAgent, VoiceAgentConfig
from core.base_graph import AgentState, AgentMode
from llm import LLMProvider
from llm.router import TaskType

logger = logging.getLogger(__name__)


# Conversation intents the agent can handle
GENERAL_INTENTS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
    "farewell": ["goodbye", "bye", "see you", "talk later", "exit", "quit"],
    "help": ["help", "what can you do", "capabilities", "features"],
    "creative": ["write", "create", "compose", "draft", "story", "poem"],
    "explain": ["explain", "what is", "how does", "tell me about", "describe"],
    "summarize": ["summarize", "summary", "brief", "tldr", "key points"],
    "brainstorm": ["brainstorm", "ideas", "suggest", "recommend", "options"],
}


class GeneralAssistant(VoiceAgent):
    """
    General-purpose conversational voice assistant.

    Optimized for natural conversation with:
    - Context-aware responses
    - Multiple conversation styles (formal, casual, creative)
    - Smart intent detection
    - Seamless voice interaction

    Default LLM: Claude (best for natural conversation)
    """

    def __init__(
        self,
        name: str = "Atlas",
        personality: str = "friendly and helpful",
        mode: AgentMode = AgentMode.VOICE,
        preferred_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize General Assistant.

        Args:
            name: Assistant's name
            personality: Personality description
            mode: Voice or text mode
            preferred_provider: Preferred LLM provider
        """
        config = VoiceAgentConfig(
            agent_name=name,
            agent_description=f"a {personality} AI assistant for general conversation",
            system_prompt=self._create_system_prompt(name, personality),
            default_provider=preferred_provider or LLMProvider.CLAUDE,
            task_type=TaskType.GENERAL,
            greeting_message=f"Hello! I'm {name}, your voice assistant. How can I help you today?",
            max_history_turns=30,  # More history for conversation continuity
        )

        super().__init__(config=config, mode=mode)

        self._personality = personality
        self._current_intent: Optional[str] = None

        logger.info(f"General Assistant '{name}' initialized")

    def _create_system_prompt(self, name: str, personality: str) -> str:
        """Create the system prompt for this assistant"""
        return f"""You are {name}, a voice-enabled AI assistant.

PERSONALITY: {personality}

CAPABILITIES:
- Natural conversation and Q&A
- Information lookup and explanations
- Creative writing (stories, poems, scripts)
- Brainstorming and idea generation
- Summarization of topics
- General assistance and recommendations

VOICE INTERACTION GUIDELINES:
1. Keep responses conversational and natural for speech
2. Avoid bullet points, numbered lists, and complex formatting
3. Use contractions and casual language when appropriate
4. Break long explanations into digestible chunks
5. Ask clarifying questions when needed
6. Be concise - aim for 2-3 sentences unless more detail is requested

IMPORTANT:
- Never mention that you're an AI unless directly asked
- Speak as if having a natural conversation
- Match the user's energy and formality level
- If you don't know something, say so honestly

Current date: {datetime.now().strftime('%A, %B %d, %Y')}
"""

    def _detect_intent(self, text: str) -> Optional[str]:
        """Detect user intent from text"""
        text_lower = text.lower()

        for intent, keywords in GENERAL_INTENTS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent

        return None

    async def process_input(self, input_data: str) -> str:
        """
        Process user input with intent detection.

        Args:
            input_data: User's message

        Returns:
            Assistant's response
        """
        # Detect intent
        self._current_intent = self._detect_intent(input_data)

        # Handle special intents
        if self._current_intent == "farewell":
            response = await self._handle_farewell()
            self._add_to_history("user", input_data)
            self._add_to_history("assistant", response)
            if self.mode == AgentMode.VOICE:
                await self.speak(response)
            return response

        if self._current_intent == "help":
            response = await self._handle_help()
            self._add_to_history("user", input_data)
            self._add_to_history("assistant", response)
            if self.mode == AgentMode.VOICE:
                await self.speak(response)
            return response

        # Adjust task type based on intent
        if self._current_intent == "creative":
            original_task_type = self.config.task_type
            self.config.task_type = TaskType.CREATIVE
            response = await super().process_input(input_data)
            self.config.task_type = original_task_type
            return response

        if self._current_intent in ["explain", "summarize"]:
            original_task_type = self.config.task_type
            self.config.task_type = TaskType.REASONING
            response = await super().process_input(input_data)
            self.config.task_type = original_task_type
            return response

        # Default processing
        return await super().process_input(input_data)

    async def _handle_farewell(self) -> str:
        """Handle farewell intent"""
        farewells = [
            "Goodbye! It was great talking with you.",
            "Take care! Feel free to come back anytime.",
            "See you later! Have a wonderful day.",
        ]
        import random
        return random.choice(farewells)

    async def _handle_help(self) -> str:
        """Handle help intent"""
        return (
            f"I'm {self.config.agent_name}, and I can help you with many things! "
            "I can answer questions, explain concepts, help with creative writing, "
            "brainstorm ideas, or just have a casual conversation. "
            "What would you like to explore?"
        )

    async def quick_response(
        self,
        question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Get a quick response without adding to conversation history.
        Useful for one-off questions.

        Args:
            question: Question to answer
            context: Optional context for the question

        Returns:
            Response text
        """
        prompt = question
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"

        # Use fast provider for quick responses
        response = await self.generate_response(
            user_message=prompt,
            system="Give a brief, direct answer. One or two sentences maximum.",
            task_type=TaskType.FAST,
        )

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def explain_topic(
        self,
        topic: str,
        depth: str = "medium"
    ) -> str:
        """
        Explain a topic at specified depth.

        Args:
            topic: Topic to explain
            depth: "brief", "medium", or "detailed"

        Returns:
            Explanation text
        """
        depth_prompts = {
            "brief": "Explain in 2-3 sentences, suitable for speaking aloud.",
            "medium": "Explain in a few paragraphs, conversationally.",
            "detailed": "Provide a comprehensive explanation, but keep it conversational.",
        }

        system = f"""Explain the following topic. {depth_prompts.get(depth, depth_prompts['medium'])}
Keep the explanation suitable for voice - no lists, bullets, or complex formatting."""

        response = await self.generate_response(
            user_message=f"Explain: {topic}",
            system=system,
            task_type=TaskType.REASONING,
        )

        # Add to history
        self._add_to_history("user", f"Explain {topic}")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def brainstorm(
        self,
        topic: str,
        count: int = 5
    ) -> str:
        """
        Brainstorm ideas on a topic.

        Args:
            topic: Topic to brainstorm about
            count: Number of ideas to generate

        Returns:
            Brainstormed ideas as natural speech
        """
        system = f"""Brainstorm {count} ideas about the given topic.
Present them conversationally, as if speaking to a friend.
Don't use numbered lists - instead, use phrases like "First...", "Another idea is...", "You could also..."
Keep it natural and easy to listen to."""

        response = await self.generate_response(
            user_message=f"Brainstorm ideas about: {topic}",
            system=system,
            task_type=TaskType.CREATIVE,
        )

        self._add_to_history("user", f"Brainstorm about {topic}")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def creative_writing(
        self,
        prompt: str,
        style: str = "narrative"
    ) -> str:
        """
        Generate creative writing.

        Args:
            prompt: Writing prompt
            style: "narrative", "poetry", "dialogue", or "descriptive"

        Returns:
            Creative text
        """
        style_guides = {
            "narrative": "Write a short narrative story.",
            "poetry": "Write a short poem.",
            "dialogue": "Write a short dialogue between characters.",
            "descriptive": "Write a vivid descriptive passage.",
        }

        system = f"""{style_guides.get(style, style_guides['narrative'])}
Keep it concise (under 200 words) and suitable for reading aloud.
Make it engaging and evocative."""

        response = await self.generate_response(
            user_message=prompt,
            system=system,
            task_type=TaskType.CREATIVE,
        )

        self._add_to_history("user", f"Write something about: {prompt}")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    def _build_graph(self) -> StateGraph:
        """Build LangGraph for general assistant"""
        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("understand", self._understand_node)
        graph.add_node("respond", self._respond_node)

        # Edges
        graph.set_entry_point("understand")
        graph.add_edge("understand", "respond")
        graph.add_edge("respond", END)

        return graph

    async def _understand_node(self, state: dict) -> dict:
        """Understand user intent"""
        task = state.get("current_task", "")
        intent = self._detect_intent(task)
        state["task_context"] = state.get("task_context", {})
        state["task_context"]["intent"] = intent
        return state

    async def _respond_node(self, state: dict) -> dict:
        """Generate response"""
        task = state.get("current_task", "")
        if task:
            response = await self.process_input(task)
            state["messages"] = state.get("messages", []) + [
                {"role": "assistant", "content": response}
            ]
        return state


# Factory function
def create_general_assistant(
    name: str = "Atlas",
    personality: str = "friendly and helpful",
    voice_mode: bool = True,
    provider: Optional[LLMProvider] = None,
) -> GeneralAssistant:
    """
    Create a General Assistant instance.

    Args:
        name: Assistant's name
        personality: Personality description
        voice_mode: Whether to enable voice mode
        provider: Preferred LLM provider

    Returns:
        Configured GeneralAssistant
    """
    return GeneralAssistant(
        name=name,
        personality=personality,
        mode=AgentMode.VOICE if voice_mode else AgentMode.TEXT,
        preferred_provider=provider,
    )
