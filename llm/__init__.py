"""
LLM Provider module for multi-model support.
Supports OpenRouter, Google Gemini, and Anthropic Claude.

NO OpenAI - using alternative providers only.
"""

from llm.provider import (
    LLMProvider,
    LLMConfig,
    BaseLLMClient,
    ClaudeClient,
    GeminiClient,
    OpenRouterClient,
)
from llm.router import LLMRouter

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "BaseLLMClient",
    "ClaudeClient",
    "GeminiClient",
    "OpenRouterClient",
    "LLMRouter",
]
