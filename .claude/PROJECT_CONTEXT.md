# langgraph-voice-agents

**Branch**: main | **Updated**: 2025-11-30

## Status
Session 2 complete rebuild with Cartesia voice and multi-LLM support. Three specialized voice agents (Atlas, Cipher, Taskmaster) with LangGraph framework. ~2,500 lines of code, no OpenAI dependencies.

## Today's Focus
1. [ ] (Add today's tasks here)

## Done (This Session)
- (none yet)

## Critical Rules
- **NO OpenAI models** - Use Cartesia for voice, Claude/Gemini/OpenRouter for text
- API keys in `.env` only, never hardcoded
- Voice-first architecture with multi-modal capabilities

## Blockers
(none)

## Quick Commands
```bash
pip install -r requirements.txt
python main.py                            # Interactive agent selection
python main.py --agent general            # Atlas - General Assistant
python main.py --agent code               # Cipher - Code Assistant
python main.py --agent task               # Taskmaster - Task Manager
python main.py --text                     # Text-only mode (no voice)
python main.py --test                     # Run integration tests
```

## Tech Stack
- **Framework**: LangGraph with BaseAgent and VoiceAgent classes
- **Voice**: Cartesia TTS (sonic-2), Cartesia STT (ink-whisper)
- **LLM Providers**: Claude (primary), Gemini (fast tasks), OpenRouter (fallback)
- **Audio**: PCM s16le, 22050 Hz TTS, 16000 Hz STT
- **Optional**: LiveKit WebRTC for multi-user voice rooms
- **Agents**: 3 specialized + 1 base class

## Recent Commits
- f889c4e feat: Session 2 - Complete rebuild with Cartesia voice and multi-LLM support
- 33e8d35 docs: add comprehensive development log and framework documentation
- 61969a9 feat: initial implementation of voice-enabled LangGraph agent framework
