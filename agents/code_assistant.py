"""
Code Assistant - Voice-enabled coding helper.

Handles:
- Code explanation and review
- Bug identification and fixes
- Code generation from descriptions
- Architecture and design discussions
- Best practices guidance

Uses Claude for code-related tasks (superior at coding).
NO OpenAI dependencies.
"""

import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END

from agents.voice_agent import VoiceAgent, VoiceAgentConfig
from core.base_graph import AgentState, AgentMode
from llm import LLMProvider
from llm.router import TaskType

logger = logging.getLogger(__name__)


class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    SQL = "sql"
    SHELL = "shell"
    UNKNOWN = "unknown"


@dataclass
class CodeContext:
    """Context for code-related discussions"""
    language: CodeLanguage = CodeLanguage.UNKNOWN
    framework: Optional[str] = None
    current_code: Optional[str] = None
    file_name: Optional[str] = None
    error_message: Optional[str] = None


class CodeAssistant(VoiceAgent):
    """
    Voice-enabled coding assistant.

    Specializes in:
    - Explaining code in natural language
    - Reviewing code for issues
    - Generating code from voice descriptions
    - Debugging and error resolution
    - Architecture discussions

    Default LLM: Claude (best for coding tasks)
    """

    def __init__(
        self,
        name: str = "Cipher",
        mode: AgentMode = AgentMode.VOICE,
        default_language: CodeLanguage = CodeLanguage.PYTHON,
    ):
        """
        Initialize Code Assistant.

        Args:
            name: Assistant's name
            mode: Voice or text mode
            default_language: Default programming language
        """
        config = VoiceAgentConfig(
            agent_name=name,
            agent_description="a coding assistant for developers",
            system_prompt=self._create_system_prompt(name),
            default_provider=LLMProvider.CLAUDE,  # Claude is best for code
            task_type=TaskType.CODING,
            greeting_message=f"Hey! I'm {name}, your coding assistant. What are we building today?",
            max_history_turns=20,
        )

        super().__init__(config=config, mode=mode)

        self._default_language = default_language
        self._code_context = CodeContext(language=default_language)

        logger.info(f"Code Assistant '{name}' initialized")

    def _create_system_prompt(self, name: str) -> str:
        """Create system prompt for code assistant"""
        return f"""You are {name}, a voice-enabled coding assistant.

EXPERTISE:
- Code explanation and review
- Bug identification and fixes
- Code generation from descriptions
- Architecture and design patterns
- Best practices and optimization
- Multiple programming languages

VOICE INTERACTION GUIDELINES:
1. When explaining code, use analogies and natural language
2. Don't read code character by character - describe what it does
3. For code generation, confirm understanding before generating
4. Keep explanations conversational but technically accurate
5. Use phrases like "this function does..." or "the bug is..."
6. When discussing code structure, paint a mental picture

CODE OUTPUT:
When generating code, format it clearly and mention:
- The language being used
- Key parts the user should pay attention to
- Any dependencies or setup needed

DEBUGGING APPROACH:
1. First understand the error or unexpected behavior
2. Ask clarifying questions if needed
3. Explain the root cause in simple terms
4. Suggest the fix with explanation

Current date: {datetime.now().strftime('%A, %B %d, %Y')}
"""

    def set_context(
        self,
        language: Optional[CodeLanguage] = None,
        framework: Optional[str] = None,
        code: Optional[str] = None,
        file_name: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """
        Set code context for more relevant responses.

        Args:
            language: Programming language
            framework: Framework being used
            code: Current code being discussed
            file_name: Name of the file
            error: Error message if debugging
        """
        if language:
            self._code_context.language = language
        if framework:
            self._code_context.framework = framework
        if code:
            self._code_context.current_code = code
        if file_name:
            self._code_context.file_name = file_name
        if error:
            self._code_context.error_message = error

        logger.debug(f"Code context updated: {self._code_context.language.value}")

    def clear_context(self):
        """Clear code context"""
        self._code_context = CodeContext(language=self._default_language)

    async def explain_code(self, code: str, language: Optional[str] = None) -> str:
        """
        Explain code in natural language.

        Args:
            code: Code to explain
            language: Optional language hint

        Returns:
            Explanation suitable for voice
        """
        self._code_context.current_code = code

        lang_hint = language or self._code_context.language.value
        system = f"""Explain the following {lang_hint} code in natural, conversational language.
Don't read the code literally - describe what it does, why it works, and any important patterns.
Keep it suitable for voice - no bullet points or complex formatting.
Start with a brief summary, then explain key parts."""

        response = await self.generate_response(
            user_message=f"Explain this code:\n```{lang_hint}\n{code}\n```",
            system=system,
            task_type=TaskType.CODING,
        )

        self._add_to_history("user", f"Explain code: {code[:100]}...")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def review_code(
        self,
        code: str,
        focus: Optional[str] = None
    ) -> str:
        """
        Review code for issues and improvements.

        Args:
            code: Code to review
            focus: Optional focus area (security, performance, style)

        Returns:
            Review feedback
        """
        self._code_context.current_code = code

        focus_text = f"Focus especially on {focus}." if focus else ""

        system = f"""Review the following code as a senior developer would.
{focus_text}
Identify:
- Any bugs or potential issues
- Security concerns
- Performance considerations
- Readability improvements

Present your review conversationally, as if discussing with a colleague.
Start with the most important issues. Be constructive and explain why each point matters."""

        response = await self.generate_response(
            user_message=f"Review this code:\n```\n{code}\n```",
            system=system,
            task_type=TaskType.CODING,
        )

        self._add_to_history("user", f"Review code: {code[:100]}...")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def generate_code(
        self,
        description: str,
        language: Optional[str] = None,
        requirements: Optional[List[str]] = None,
    ) -> str:
        """
        Generate code from natural language description.

        Args:
            description: What the code should do
            language: Target programming language
            requirements: Additional requirements

        Returns:
            Generated code with explanation
        """
        lang = language or self._code_context.language.value
        req_text = ""
        if requirements:
            req_text = f"\nAdditional requirements: {', '.join(requirements)}"

        system = f"""Generate {lang} code based on the user's description.
{req_text}

First, briefly confirm your understanding of what's needed (1 sentence).
Then provide the code with a brief explanation of how it works.
Keep explanations conversational for voice.
Make the code clean, well-structured, and following best practices."""

        response = await self.generate_response(
            user_message=f"Generate code: {description}",
            system=system,
            task_type=TaskType.CODING,
        )

        # Extract code for context
        code_match = re.search(r'```[\w]*\n(.*?)```', response, re.DOTALL)
        if code_match:
            self._code_context.current_code = code_match.group(1)

        self._add_to_history("user", f"Generate: {description}")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            # For voice, just speak the explanation part, not the code
            explanation = re.sub(r'```[\w]*\n.*?```', '', response, flags=re.DOTALL).strip()
            if explanation:
                await self.speak(explanation + " I've generated the code for you to review.")

        return response

    async def debug_error(
        self,
        error_message: str,
        code: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Help debug an error.

        Args:
            error_message: The error message
            code: Code that caused the error
            context: Additional context

        Returns:
            Debugging help
        """
        self._code_context.error_message = error_message
        if code:
            self._code_context.current_code = code

        context_text = f"\nContext: {context}" if context else ""
        code_text = f"\nCode:\n```\n{code}\n```" if code else ""

        system = """Help debug this error like a patient senior developer.
1. Explain what the error means in simple terms
2. Identify the likely cause
3. Suggest a specific fix
4. Explain why this fix works

Be conversational and reassuring - debugging is normal and everyone encounters errors."""

        response = await self.generate_response(
            user_message=f"Error: {error_message}{code_text}{context_text}",
            system=system,
            task_type=TaskType.CODING,
        )

        self._add_to_history("user", f"Debug error: {error_message}")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def explain_concept(self, concept: str) -> str:
        """
        Explain a programming concept.

        Args:
            concept: Concept to explain

        Returns:
            Explanation
        """
        system = """Explain this programming concept clearly and conversationally.
Use analogies to real-world things when helpful.
Include a simple example if it aids understanding.
Keep it suitable for voice - no complex formatting."""

        response = await self.generate_response(
            user_message=f"Explain the concept: {concept}",
            system=system,
            task_type=TaskType.CODING,
        )

        self._add_to_history("user", f"Explain concept: {concept}")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def suggest_architecture(
        self,
        project_description: str,
        constraints: Optional[List[str]] = None,
    ) -> str:
        """
        Suggest architecture for a project.

        Args:
            project_description: What the project should do
            constraints: Technical constraints

        Returns:
            Architecture suggestions
        """
        constraints_text = ""
        if constraints:
            constraints_text = f"\nConstraints: {', '.join(constraints)}"

        system = """Suggest architecture for this project as a senior architect would.
Discuss:
- High-level structure and components
- Key technology choices and why
- Important patterns to use
- Potential challenges to consider

Keep it conversational and explain your reasoning.
Don't use bullet points - speak naturally about the architecture."""

        response = await self.generate_response(
            user_message=f"Project: {project_description}{constraints_text}",
            system=system,
            task_type=TaskType.REASONING,
        )

        self._add_to_history("user", f"Architecture for: {project_description}")
        self._add_to_history("assistant", response)

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def process_input(self, input_data: str) -> str:
        """
        Process code-related input with context awareness.

        Args:
            input_data: User's message

        Returns:
            Response
        """
        # Detect if discussing code
        input_lower = input_data.lower()

        # Auto-detect language mentions
        for lang in CodeLanguage:
            if lang.value in input_lower:
                self._code_context.language = lang
                break

        # Check for error-related queries
        if any(word in input_lower for word in ["error", "bug", "crash", "exception", "failing"]):
            if "fix" in input_lower or "help" in input_lower:
                return await self.debug_error(input_data)

        # Check for explanation requests
        if any(word in input_lower for word in ["explain", "what does", "how does", "what is"]):
            if "code" in input_lower and self._code_context.current_code:
                return await self.explain_code(self._code_context.current_code)

        # Default processing
        return await super().process_input(input_data)

    def _build_graph(self) -> StateGraph:
        """Build LangGraph for code assistant"""
        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("execute", self._execute_node)

        # Edges
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "execute")
        graph.add_edge("execute", END)

        return graph

    async def _analyze_node(self, state: dict) -> dict:
        """Analyze the coding request"""
        task = state.get("current_task", "")

        # Determine request type
        context = state.get("task_context", {})

        if "explain" in task.lower():
            context["request_type"] = "explain"
        elif "review" in task.lower():
            context["request_type"] = "review"
        elif "generate" in task.lower() or "create" in task.lower() or "write" in task.lower():
            context["request_type"] = "generate"
        elif "debug" in task.lower() or "error" in task.lower():
            context["request_type"] = "debug"
        else:
            context["request_type"] = "general"

        state["task_context"] = context
        return state

    async def _execute_node(self, state: dict) -> dict:
        """Execute the coding task"""
        task = state.get("current_task", "")
        if task:
            response = await self.process_input(task)
            state["messages"] = state.get("messages", []) + [
                {"role": "assistant", "content": response}
            ]
        return state


# Factory function
def create_code_assistant(
    name: str = "Cipher",
    voice_mode: bool = True,
    default_language: str = "python",
) -> CodeAssistant:
    """
    Create a Code Assistant instance.

    Args:
        name: Assistant's name
        voice_mode: Whether to enable voice mode
        default_language: Default programming language

    Returns:
        Configured CodeAssistant
    """
    lang = CodeLanguage.UNKNOWN
    try:
        lang = CodeLanguage(default_language.lower())
    except ValueError:
        pass

    return CodeAssistant(
        name=name,
        mode=AgentMode.VOICE if voice_mode else AgentMode.TEXT,
        default_language=lang,
    )
