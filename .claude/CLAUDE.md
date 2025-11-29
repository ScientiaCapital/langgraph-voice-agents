# CLAUDE.md - LangGraph Voice Agents Project Guide

## Project Status & Overview

**Current Status**: Session 2 Complete (2025-01-28)
**Type**: Voice-enabled Multi-Agent AI Framework
**Focus**: Real-time voice interactions with AI agents using LangGraph orchestration

**CRITICAL: NO OpenAI dependencies in this project.**

This project implements a voice-enabled agent system that combines:
- **LangGraph** for stateful agent workflows and orchestration
- **Cartesia** for TTS (sonic-2) and STT (ink-whisper)
- **LiveKit** for optional WebRTC voice transport
- **Multi-LLM** support via Claude, Gemini, and OpenRouter

## Technology Stack

### Core Frameworks & Libraries
```python
# Primary Dependencies
langgraph >= 0.1.0          # Agent workflow orchestration
langchain >= 0.1.0          # LLM integration and tooling
langchain-core >= 0.1.0     # Core LangChain components
cartesia >= 1.0.0           # Voice TTS/STT (NO OpenAI)
livekit >= 0.11.0           # Real-time voice/audio transport

# LLM Providers (NO OpenAI)
anthropic >= 0.25.0         # Claude
google-generativeai >= 0.5.0 # Gemini
httpx >= 0.27.0             # OpenRouter API
```

### Architecture Components
- **State Management**: LangGraph StateGraph with typed state objects
- **Voice Processing**: Cartesia TTS/STT + optional LiveKit WebRTC
- **LLM Routing**: Task-based provider selection with failover
- **Multi-Modal Support**: Simultaneous text and voice interaction modes

## Development Workflow

### Initial Setup
```bash
# Clone and setup
git clone <repository>
cd langgraph-voice-agents

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application
```bash
# Interactive mode - choose an agent
python main.py

# Start specific agent
python main.py --agent general   # Atlas - General Assistant
python main.py --agent code      # Cipher - Code Assistant
python main.py --agent task      # Taskmaster - Task Manager

# Text-only mode (no voice)
python main.py --text

# Run integration tests
python main.py --test
```

## Environment Variables

### Required (at least one LLM)
```bash
ANTHROPIC_API_KEY=sk-ant-...      # Claude
GOOGLE_API_KEY=AIza...            # Gemini
OPENROUTER_API_KEY=sk-or-...      # OpenRouter
```

### Voice (required for voice mode)
```bash
CARTESIA_API_KEY=...              # Cartesia TTS/STT
```

### Optional (LiveKit WebRTC)
```bash
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

## Key Files & Their Purposes

### Project Structure
```
langgraph-voice-agents/
├── main.py                          # CLI entry point
├── agents/                          # Voice-enabled agents
│   ├── __init__.py                  # Module exports
│   ├── voice_agent.py               # Base VoiceAgent class
│   ├── general_assistant.py         # Atlas - conversation
│   ├── code_assistant.py            # Cipher - coding help
│   └── task_manager.py              # Taskmaster - productivity
├── llm/                             # Multi-LLM support
│   ├── __init__.py                  # Module exports
│   ├── provider.py                  # Claude, Gemini, OpenRouter clients
│   └── router.py                    # Task-based provider routing
├── voice/                           # Voice processing
│   ├── __init__.py                  # Module exports
│   ├── cartesia_client.py           # Cartesia TTS/STT
│   ├── livekit_client.py            # LiveKit + Cartesia integration
│   └── audio_utils.py               # PCM/WAV conversion
├── core/                            # Infrastructure
│   ├── __init__.py                  # Module exports
│   ├── base_graph.py                # BaseAgent, mixins
│   └── state_management.py          # SQLite/Redis persistence
├── requirements.txt
├── .env.example
├── CLAUDE.md                        # Development log
└── .claude/
    ├── context.md                   # Current state
    └── CLAUDE.md                    # This file
```

### Critical Implementation Files
- **`agents/voice_agent.py`**: Base class with voice + LLM integration
- **`llm/router.py`**: Task-based LLM provider selection
- **`voice/cartesia_client.py`**: Cartesia TTS/STT WebSocket client
- **`main.py`**: CLI with environment check and agent selection

## Coding Standards

### LangGraph Patterns
```python
# State definition using dataclass
@dataclass
class AgentState:
    session_id: str
    agent_type: str
    mode: AgentMode
    current_task: Optional[str] = None
    messages: List[Dict[str, Any]] = None
```

### Voice Integration Standards
```python
# Cartesia voice processing pattern
class VoiceAgent(MultiModalMixin, LLMIntegrationMixin, BaseAgent):
    async def speak(self, text: str) -> bool:
        """Synthesize and speak text via Cartesia."""
        async for chunk in self._cartesia_client.speak(text):
            yield chunk
```

### LLM Provider Selection
```python
# Task-based routing
PROVIDER_RANKINGS = {
    TaskType.GENERAL: [CLAUDE, GEMINI, OPENROUTER],
    TaskType.CODING: [CLAUDE, OPENROUTER, GEMINI],
    TaskType.FAST: [GEMINI, OPENROUTER, CLAUDE],
}
```

## Common Tasks & Commands

### Development Tasks
```bash
# Run with debug logging
python main.py --debug

# Test specific agent in text mode
python main.py --agent code --text

# Verify environment
python main.py --test
```

### Agent Interaction Patterns
```python
# Example: Create and use General Assistant
from agents import create_general_assistant

agent = create_general_assistant(voice_mode=True)
await agent.start_voice_session()
response = await agent.process_input("Hello, how are you?")
await agent.close()
```

## Troubleshooting Tips

### Common Issues & Solutions

**Cartesia Connection Failures**
```python
# Check Cartesia configuration
import os
print(f"Cartesia key set: {bool(os.getenv('CARTESIA_API_KEY'))}")
```

**No LLM Providers Available**
```bash
# Run environment check
python main.py --test
```

**Voice Not Working**
- Verify `CARTESIA_API_KEY` is set
- Run with `--text` to confirm LLM is working
- Check audio format compatibility (PCM s16le)

### Performance Tips
- Use Gemini for fast responses (`TaskType.FAST`)
- Use Claude for complex reasoning and coding
- Stream responses for lower perceived latency

## Session History

| Session | Date | Summary |
|---------|------|---------|
| 1 | 2025-01-21 | Initial implementation with MCP tools and OpenAI voice |
| 2 | 2025-01-28 | Complete rebuild: Cartesia voice, multi-LLM, no OpenAI |

---
*Last Updated: 2025-01-28*
