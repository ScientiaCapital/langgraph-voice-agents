# LangGraph Voice Agents - Project Context

**Last Updated:** 2025-01-28 - Session 2 Complete

## Current Sprint Focus
- ✅ **Voice Integration**: Cartesia TTS/STT + LiveKit WebRTC
- ✅ **Multi-LLM Support**: Claude, Gemini, OpenRouter (NO OpenAI)
- ✅ **Voice-First Agents**: General Assistant, Code Assistant, Task Manager
- ✅ **CLI Entry Point**: Interactive agent selection and testing

## Architecture Overview
- **Language**: Python 3.11+
- **Framework**: LangGraph for agent orchestration
- **Type**: Voice-enabled multi-agent system
- **Voice Provider**: Cartesia (TTS: sonic-2, STT: ink-whisper)
- **LLM Providers**: Claude, Gemini, OpenRouter
- **Transport**: LiveKit WebRTC (optional)

## Project Structure
```
langgraph-voice-agents/
├── main.py                 # CLI entry point
├── agents/                 # Voice-enabled agents
│   ├── voice_agent.py      # Base VoiceAgent class
│   ├── general_assistant.py # Atlas - conversation
│   ├── code_assistant.py   # Cipher - coding help
│   └── task_manager.py     # Taskmaster - productivity
├── llm/                    # Multi-LLM support
│   ├── provider.py         # Claude, Gemini, OpenRouter
│   └── router.py           # Task-based routing
├── voice/                  # Voice processing
│   ├── cartesia_client.py  # Cartesia TTS/STT
│   ├── livekit_client.py   # LiveKit + Cartesia
│   └── audio_utils.py      # PCM/WAV conversion
├── core/                   # Infrastructure
│   ├── base_graph.py       # BaseAgent, mixins
│   └── state_management.py # SQLite/Redis
├── requirements.txt
├── .env.example
└── CLAUDE.md
```

## Session 2 Changes (2025-01-28)
- **Removed**: All OpenAI dependencies, `/tools/` directory, old MCP-based agents
- **Added**: Cartesia voice, multi-LLM router, three new specialized agents
- **Modified**: `base_graph.py` with new mixins, `livekit_client.py` for Cartesia

## Critical Rules
- **NO OpenAI** - Use Claude/Gemini/OpenRouter for LLM, Cartesia for voice
- **Voice-First** - All agents support voice interaction
- **Graceful Degradation** - Falls back to text if voice unavailable

## Recent Changes
| File | Change |
|------|--------|
| `agents/` | New voice agents: Atlas, Cipher, Taskmaster |
| `llm/` | New module: multi-provider with routing |
| `voice/` | Cartesia integration, updated LiveKit |
| `core/base_graph.py` | Added LLMIntegrationMixin |
| `main.py` | New CLI with agent selection |

## Current Blockers
- None - Session 2 implementation complete

## Next Steps
1. **Test with real API keys** - Verify Cartesia and LLM integrations work
2. **Add unit tests** - Test voice and LLM components
3. **Enhance agents** - Add more specialized behaviors
4. **LiveKit testing** - Test WebRTC room functionality
5. **Production hardening** - Error handling, logging, monitoring

## Environment Requirements
```bash
# Required (at least one LLM)
ANTHROPIC_API_KEY=...    # Claude
GOOGLE_API_KEY=...       # Gemini
OPENROUTER_API_KEY=...   # OpenRouter

# Required for voice
CARTESIA_API_KEY=...

# Optional (LiveKit)
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

## Quick Commands
```bash
# Run interactively
python main.py

# Specific agent
python main.py --agent general
python main.py --agent code
python main.py --agent task

# Text mode (no voice)
python main.py --text

# Run tests
python main.py --test
```

## Notes
- All three agents are fully implemented and ready for testing
- The old MCP tool adapters were removed to start fresh
- LiveKit is optional - agents work with just Cartesia for voice
- Task-based LLM routing selects optimal provider automatically
