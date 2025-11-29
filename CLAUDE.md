# LangGraph Voice-Enabled Agent Framework - Development Log

## 🎯 Project Overview
Voice-enabled multi-agent framework built on LangGraph with Cartesia voice integration and multi-LLM support.

**Critical Rule: NO OpenAI dependencies anywhere in this project.**

## ✅ Session 2 - Complete Rebuild (2025-01-28)

### Major Changes
- **Removed all OpenAI dependencies** - Replaced with Cartesia for voice, multi-LLM for text
- **Deleted old MCP tool adapters** - Starting fresh with voice-first agents
- **New agent architecture** - Three specialized voice agents

### Voice Module (`/voice/`)
| File | Purpose |
|------|---------|
| `audio_utils.py` | PCM/WAV conversion, f32↔s16, RMS amplitude calculation |
| `cartesia_client.py` | Cartesia TTS (sonic-2) and STT (ink-whisper) via WebSocket |
| `livekit_client.py` | LiveKit WebRTC + Cartesia integration for real-time rooms |
| `__init__.py` | Module exports |

### LLM Module (`/llm/`)
| File | Purpose |
|------|---------|
| `provider.py` | Claude, Gemini, OpenRouter client implementations |
| `router.py` | Task-based provider selection with automatic failover |
| `__init__.py` | Module exports |

### Core Module (`/core/`)
| File | Purpose |
|------|---------|
| `base_graph.py` | BaseAgent, AgentState, MultiModalMixin, LLMIntegrationMixin |
| `state_management.py` | SQLite/Redis hybrid state persistence |
| `__init__.py` | Module exports |

### Agents (`/agents/`)
| Agent | Name | Specialization | Default LLM |
|-------|------|----------------|-------------|
| `general_assistant.py` | Atlas | Conversation, Q&A, creative writing | Claude |
| `code_assistant.py` | Cipher | Code explanation, review, generation | Claude |
| `task_manager.py` | Taskmaster | Task tracking, project breakdown | Gemini (fast) |
| `voice_agent.py` | Base class | Voice + LLM capabilities | Configurable |

### Entry Point
- `main.py` - CLI with interactive agent selection, environment check, integration tests

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     agents/ Module                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ General     │  │ Code        │  │ Task Manager        │ │
│  │ Assistant   │  │ Assistant   │  │ (Taskmaster)        │ │
│  │ (Atlas)     │  │ (Cipher)    │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └────────────────┼─────────────────────┘           │
│                          ▼                                  │
│              ┌───────────────────────┐                     │
│              │    VoiceAgent Base    │                     │
│              │ (voice_agent.py)      │                     │
│              └───────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   core/ Module  │  │   llm/ Module   │  │  voice/ Module  │
│                 │  │                 │  │                 │
│ • BaseAgent     │  │ • ClaudeClient  │  │ • CartesiaTTS   │
│ • AgentState    │  │ • GeminiClient  │  │ • CartesiaSTT   │
│ • MultiModal    │  │ • OpenRouter    │  │ • LiveKit       │
│   Mixin         │  │ • LLMRouter     │  │   Integration   │
│ • LLMIntegration│  │                 │  │                 │
│   Mixin         │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 🔧 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - CARTESIA_API_KEY (required for voice)
# - ANTHROPIC_API_KEY (Claude)
# - GOOGLE_API_KEY (Gemini)
# - OPENROUTER_API_KEY (OpenRouter)
# - LIVEKIT_* (optional, for WebRTC rooms)

# Run interactively
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

## 📋 Environment Variables

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

## 🎙️ Voice Features

### Cartesia Integration
- **TTS Model**: `sonic-2` - High-quality voice synthesis
- **STT Model**: `ink-whisper` - Accurate transcription
- **Sample Rate**: 22050 Hz (TTS), 16000 Hz (STT)
- **Format**: PCM s16le for playback compatibility

### Voice Activity Detection
- RMS amplitude-based speech detection
- Configurable silence timeout (1.5s default)
- Minimum speech duration filtering (0.3s)

### LiveKit Support (Optional)
- WebRTC rooms for multi-user voice sessions
- Real-time audio streaming
- Participant management

## 🧠 LLM Provider Rankings

| Task Type | 1st Choice | 2nd Choice | 3rd Choice |
|-----------|------------|------------|------------|
| General | Claude | Gemini | OpenRouter |
| Coding | Claude | OpenRouter | Gemini |
| Reasoning | Claude | OpenRouter | Gemini |
| Fast | Gemini | OpenRouter | Claude |
| Creative | Claude | Gemini | OpenRouter |

## 📊 Project Metrics

- **Python Files**: 15
- **Lines of Code**: ~2,500
- **Agents**: 3 specialized + 1 base class
- **LLM Providers**: 3 (Claude, Gemini, OpenRouter)
- **Voice Provider**: 1 (Cartesia)

## 🚫 What Was Removed

Session 2 removed the following from Session 1:
- `/tools/` directory (MCP tool adapters)
- Old agents: `task_orchestrator.py`, `task_executor.py`, `task_checker.py`
- OpenAI Whisper integration
- OpenAI TTS integration

## 📝 Session History

| Session | Date | Summary |
|---------|------|---------|
| 1 | 2025-01-21 | Initial implementation with MCP tools and OpenAI voice |
| 2 | 2025-01-28 | Complete rebuild with Cartesia voice, multi-LLM, no OpenAI |

---
*Last Updated: 2025-01-28 - Session 2 Complete*
