"""
LLM Provider implementations for voice agents.
Supports Anthropic Claude, Google Gemini, and OpenRouter.

NO OpenAI - these are the exclusive LLM providers.
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, AsyncGenerator, List, Dict, Any

import anthropic
import google.generativeai as genai
import httpx

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers"""
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


@dataclass
class Message:
    """Chat message structure"""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    # API Keys (default to environment variables)
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Model defaults
    claude_model: str = "claude-sonnet-4-5-20250929"
    gemini_model: str = "gemini-1.5-flash"
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    # Generation settings
    max_tokens: int = 4096
    temperature: float = 0.7

    def __post_init__(self):
        # Load from environment if not provided
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.google_api_key is None:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider type"""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available"""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a complete response"""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens"""
        pass


class ClaudeClient(BaseLLMClient):
    """
    Anthropic Claude client.
    Highest quality responses, best for complex reasoning.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Optional[anthropic.AsyncAnthropic] = None

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.CLAUDE

    @property
    def is_available(self) -> bool:
        return bool(self.config.anthropic_api_key)

    def _ensure_client(self):
        if self._client is None:
            if not self.is_available:
                raise ValueError("Anthropic API key not configured")
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.anthropic_api_key
            )

    async def generate(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate complete response from Claude"""
        self._ensure_client()

        # Convert messages to Anthropic format
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"  # System handled separately
        ]

        response = await self._client.messages.create(
            model=kwargs.get("model", self.config.claude_model),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            system=system or "",
            messages=anthropic_messages
        )

        return response.content[0].text

    async def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from Claude"""
        self._ensure_client()

        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        async with self._client.messages.stream(
            model=kwargs.get("model", self.config.claude_model),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            system=system or "",
            messages=anthropic_messages
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GeminiClient(BaseLLMClient):
    """
    Google Gemini client.
    Fast and cost-effective, good for general tasks.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model = None
        self._initialized = False

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.GEMINI

    @property
    def is_available(self) -> bool:
        return bool(self.config.google_api_key)

    def _ensure_client(self):
        if not self._initialized:
            if not self.is_available:
                raise ValueError("Google API key not configured")
            genai.configure(api_key=self.config.google_api_key)
            self._model = genai.GenerativeModel(self.config.gemini_model)
            self._initialized = True

    async def generate(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate complete response from Gemini"""
        self._ensure_client()

        # Build conversation history
        history = []
        for msg in messages[:-1]:  # All but last message
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})

        # Start chat with history
        chat = self._model.start_chat(history=history)

        # Build prompt with system message if provided
        last_message = messages[-1].content if messages else ""
        if system:
            prompt = f"System: {system}\n\nUser: {last_message}"
        else:
            prompt = last_message

        # Generate response
        response = await asyncio.to_thread(
            chat.send_message,
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
            )
        )

        return response.text

    async def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from Gemini"""
        self._ensure_client()

        # Build conversation
        history = []
        for msg in messages[:-1]:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})

        chat = self._model.start_chat(history=history)

        last_message = messages[-1].content if messages else ""
        if system:
            prompt = f"System: {system}\n\nUser: {last_message}"
        else:
            prompt = last_message

        # Stream response
        response = await asyncio.to_thread(
            chat.send_message,
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
            ),
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter client.
    Access to many models through a single API.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENROUTER

    @property
    def is_available(self) -> bool:
        return bool(self.config.openrouter_api_key)

    def _ensure_client(self):
        if self._client is None:
            if not self.is_available:
                raise ValueError("OpenRouter API key not configured")
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.config.openrouter_api_key}",
                    "HTTP-Referer": "https://github.com/langgraph-voice-agents",
                    "X-Title": "LangGraph Voice Agents"
                },
                timeout=60.0
            )

    async def generate(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate complete response from OpenRouter"""
        self._ensure_client()

        # Build messages in OpenAI format
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": kwargs.get("model", self.config.openrouter_model),
                "messages": api_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from OpenRouter"""
        self._ensure_client()

        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        async with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": kwargs.get("model", self.config.openrouter_model),
                "messages": api_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
