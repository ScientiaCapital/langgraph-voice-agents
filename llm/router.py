"""
LLM Router for intelligent provider selection.
Routes requests to the best available provider based on task and availability.
"""

import logging
from typing import Optional, AsyncGenerator, List, Dict
from enum import Enum

from llm.provider import (
    LLMProvider,
    LLMConfig,
    BaseLLMClient,
    ClaudeClient,
    GeminiClient,
    OpenRouterClient,
    Message,
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks for provider selection"""
    GENERAL = "general"         # General conversation
    CODING = "coding"           # Code generation/explanation
    REASONING = "reasoning"     # Complex reasoning tasks
    FAST = "fast"               # Quick responses needed
    CREATIVE = "creative"       # Creative writing


# Provider rankings by task type
PROVIDER_RANKINGS: Dict[TaskType, List[LLMProvider]] = {
    TaskType.GENERAL: [
        LLMProvider.CLAUDE,
        LLMProvider.GEMINI,
        LLMProvider.OPENROUTER
    ],
    TaskType.CODING: [
        LLMProvider.CLAUDE,
        LLMProvider.OPENROUTER,
        LLMProvider.GEMINI
    ],
    TaskType.REASONING: [
        LLMProvider.CLAUDE,
        LLMProvider.OPENROUTER,
        LLMProvider.GEMINI
    ],
    TaskType.FAST: [
        LLMProvider.GEMINI,
        LLMProvider.OPENROUTER,
        LLMProvider.CLAUDE
    ],
    TaskType.CREATIVE: [
        LLMProvider.CLAUDE,
        LLMProvider.GEMINI,
        LLMProvider.OPENROUTER
    ],
}


class LLMRouter:
    """
    Intelligent router for LLM providers.

    Features:
    - Automatic failover between providers
    - Task-based provider selection
    - Caching of provider availability
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

        # Initialize all providers
        self._clients: Dict[LLMProvider, BaseLLMClient] = {
            LLMProvider.CLAUDE: ClaudeClient(self.config),
            LLMProvider.GEMINI: GeminiClient(self.config),
            LLMProvider.OPENROUTER: OpenRouterClient(self.config),
        }

        # Track which providers are available
        self._availability = {
            provider: client.is_available
            for provider, client in self._clients.items()
        }

        logger.info(f"LLM Router initialized. Available providers: {self.available_providers}")

    @property
    def available_providers(self) -> List[LLMProvider]:
        """List of providers that are currently available"""
        return [p for p, available in self._availability.items() if available]

    def get_client(self, provider: LLMProvider) -> BaseLLMClient:
        """Get a specific provider's client"""
        if not self._availability.get(provider, False):
            raise ValueError(f"Provider {provider.value} is not available")
        return self._clients[provider]

    def select_provider(
        self,
        task_type: TaskType = TaskType.GENERAL,
        preferred: Optional[LLMProvider] = None
    ) -> LLMProvider:
        """
        Select the best available provider for a task.

        Args:
            task_type: Type of task to perform
            preferred: Preferred provider (used if available)

        Returns:
            Selected provider
        """
        # Use preferred if available
        if preferred and self._availability.get(preferred, False):
            return preferred

        # Find best available provider for task type
        rankings = PROVIDER_RANKINGS.get(task_type, PROVIDER_RANKINGS[TaskType.GENERAL])
        for provider in rankings:
            if self._availability.get(provider, False):
                return provider

        raise ValueError("No LLM providers are available")

    async def generate(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> str:
        """
        Generate a complete response.

        Args:
            messages: Conversation messages
            system: System prompt
            provider: Specific provider to use (optional)
            task_type: Type of task for automatic provider selection
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response text
        """
        selected = self.select_provider(task_type, provider)
        client = self._clients[selected]

        logger.debug(f"Using {selected.value} for generation")

        try:
            return await client.generate(messages, system, **kwargs)
        except Exception as e:
            logger.error(f"Generation failed with {selected.value}: {e}")

            # Try fallback providers
            for fallback in self.available_providers:
                if fallback != selected:
                    logger.info(f"Falling back to {fallback.value}")
                    try:
                        return await self._clients[fallback].generate(
                            messages, system, **kwargs
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback {fallback.value} also failed: {fallback_error}")

            raise RuntimeError("All LLM providers failed")

    async def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens.

        Args:
            messages: Conversation messages
            system: System prompt
            provider: Specific provider to use (optional)
            task_type: Type of task for automatic provider selection
            **kwargs: Additional provider-specific parameters

        Yields:
            Response tokens as they're generated
        """
        selected = self.select_provider(task_type, provider)
        client = self._clients[selected]

        logger.debug(f"Using {selected.value} for streaming")

        try:
            async for token in client.stream(messages, system, **kwargs):
                yield token
        except Exception as e:
            logger.error(f"Streaming failed with {selected.value}: {e}")
            # For streaming, we don't do fallback as it would break the stream
            raise

    async def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Message]] = None,
        system: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> str:
        """
        Convenience method for single-turn chat.

        Args:
            user_message: User's message
            conversation_history: Previous messages (optional)
            system: System prompt
            provider: Specific provider
            task_type: Task type for provider selection
            **kwargs: Additional parameters

        Returns:
            Assistant's response
        """
        messages = list(conversation_history) if conversation_history else []
        messages.append(Message(role="user", content=user_message))

        return await self.generate(
            messages=messages,
            system=system,
            provider=provider,
            task_type=task_type,
            **kwargs
        )

    async def stream_chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Message]] = None,
        system: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Convenience method for single-turn streaming chat.

        Args:
            user_message: User's message
            conversation_history: Previous messages (optional)
            system: System prompt
            provider: Specific provider
            task_type: Task type for provider selection
            **kwargs: Additional parameters

        Yields:
            Response tokens
        """
        messages = list(conversation_history) if conversation_history else []
        messages.append(Message(role="user", content=user_message))

        async for token in self.stream(
            messages=messages,
            system=system,
            provider=provider,
            task_type=task_type,
            **kwargs
        ):
            yield token

    async def close(self):
        """Close all provider connections"""
        for client in self._clients.values():
            if hasattr(client, 'close'):
                await client.close()
