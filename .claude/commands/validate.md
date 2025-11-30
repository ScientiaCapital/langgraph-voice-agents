---
description: "Multi-phase validation for langgraph-voice-agents"
---

# Validate langgraph-voice-agents

## Critical Rules
- **NO OpenAI** - Use Claude, Gemini, OpenRouter only
- **Cartesia for all voice** - TTS (sonic-2) and STT (ink-whisper)
- API keys in `.env` only, never hardcoded
- At least one LLM provider required

---

## Phase 1: Environment Check

Run the built-in environment verification:

```bash
cd /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents
python main.py --test
```

**Expected Checks:**
- ✅ At least one LLM provider configured (Claude/Gemini/OpenRouter)
- ✅ Cartesia API key set (for voice features)
- ⚠️ LiveKit optional (WebRTC rooms)

**Required Environment Variables:**
```bash
# At least ONE of these (LLM):
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...

# Required for voice mode:
CARTESIA_API_KEY=...

# Optional (LiveKit WebRTC):
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

---

## Phase 2: Code Quality

### Formatting Check
```bash
black --check /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents
```

### Type Checking
```bash
mypy /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents --ignore-missing-imports
```

### Linting (Optional)
```bash
ruff check /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents
```

---

## Phase 3: Module Integration Tests

### Test 1: LLM Router
Verify multi-LLM routing with automatic failover:

```bash
python -c "
import asyncio
from llm import LLMConfig, LLMRouter

async def test():
    router = LLMRouter(LLMConfig())
    print(f'Available providers: {router.available_providers}')

    if router.available_providers:
        response = await router.chat('Say hello')
        print(f'Response: {response[:50]}...')
        await router.close()
    else:
        print('ERROR: No LLM providers available')

asyncio.run(test())
"
```

### Test 2: Cartesia Voice Client
Verify TTS and STT configuration:

```bash
python -c "
import asyncio
from voice import CartesiaConfig

async def test():
    config = CartesiaConfig()
    print(f'TTS Model: {config.tts_model}')  # Should be 'sonic-2'
    print(f'STT Model: {config.stt_model}')  # Should be 'ink-whisper'
    print(f'Sample Rate: {config.sample_rate}')

asyncio.run(test())
"
```

### Test 3: Agent Creation
Verify all three specialized agents can be created:

```bash
python -c "
import asyncio
from agents import (
    create_general_assistant,
    create_code_assistant,
    create_task_manager
)

async def test():
    atlas = create_general_assistant(voice_mode=False)
    print(f'✅ Created: {atlas.config.agent_name}')
    await atlas.close()

    cipher = create_code_assistant(voice_mode=False)
    print(f'✅ Created: {cipher.config.agent_name}')
    await cipher.close()

    taskmaster = create_task_manager(voice_mode=False)
    print(f'✅ Created: {taskmaster.config.agent_name}')
    await taskmaster.close()

asyncio.run(test())
"
```

---

## Phase 4: Voice Module Check

### Cartesia Integration Checklist
- [ ] TTS model configured: `sonic-2`
- [ ] STT model configured: `ink-whisper`
- [ ] Audio utils: PCM/WAV conversion functions
- [ ] Sample rate: 22050 Hz (TTS), 16000 Hz (STT)
- [ ] Format: PCM s16le for playback

### Voice Activity Detection
- [ ] RMS amplitude-based speech detection
- [ ] Configurable silence timeout (1.5s default)
- [ ] Minimum speech duration filtering (0.3s)

### LiveKit (Optional)
- [ ] WebRTC room support
- [ ] Real-time audio streaming
- [ ] Participant management

---

## Phase 5: Architecture Validation

### Module Structure
```
/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/
├── voice/              # Cartesia TTS/STT, LiveKit
│   ├── audio_utils.py
│   ├── cartesia_client.py
│   └── livekit_client.py
├── llm/                # Claude, Gemini, OpenRouter
│   ├── provider.py
│   └── router.py
├── core/               # Base agent, state management
│   ├── base_graph.py
│   └── state_management.py
├── agents/             # Specialized agents
│   ├── voice_agent.py      # Base class
│   ├── general_assistant.py # Atlas
│   ├── code_assistant.py    # Cipher
│   └── task_manager.py      # Taskmaster
└── main.py             # CLI entry point
```

### Agent Hierarchy
```
BaseAgent
  └── MultiModalMixin + LLMIntegrationMixin + ErrorHandlingMixin
      └── VoiceAgent (base class)
          ├── Atlas (General Assistant)
          ├── Cipher (Code Assistant)
          └── Taskmaster (Task Manager)
```

---

## Phase 6: Dependency Verification

### Critical: NO OpenAI Dependencies
```bash
# This should return EMPTY - no OpenAI packages
grep -i "openai" /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/requirements.txt
```

### Required Dependencies
```bash
# Check key packages are installed
pip list | grep -E "(langgraph|cartesia|anthropic|google-generativeai|livekit)"
```

**Expected Output:**
- `langgraph` >= 0.2.0
- `cartesia` >= 1.0.0
- `anthropic` >= 0.25.0 (optional)
- `google-generativeai` >= 0.5.0 (optional)
- `livekit` >= 0.11.0 (optional)

---

## Phase 7: Interactive Test

Start an agent and verify voice/text mode switching:

```bash
python /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/main.py --agent general
```

**Commands to test:**
- `voice on` - Enable voice mode
- `voice off` - Disable voice mode
- `clear` - Clear conversation history
- `quit` - Exit

---

## Success Criteria

✅ All phases pass without errors
✅ At least one LLM provider available
✅ Cartesia voice configured (if using voice mode)
✅ No OpenAI dependencies found
✅ All three agents (Atlas, Cipher, Taskmaster) create successfully
✅ LLM router handles failover between providers
✅ Voice module uses only Cartesia (sonic-2 TTS, ink-whisper STT)

---

## Common Issues

### Issue: No LLM providers available
**Fix:** Set at least one API key in `.env`:
- `ANTHROPIC_API_KEY` (Claude)
- `GOOGLE_API_KEY` (Gemini)
- `OPENROUTER_API_KEY` (OpenRouter)

### Issue: Voice not working
**Fix:** Set `CARTESIA_API_KEY` in `.env`

### Issue: Import errors
**Fix:** Install dependencies:
```bash
pip install -r /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/requirements.txt
```

### Issue: LiveKit errors
**Fix:** LiveKit is optional. Either configure it or ignore WebRTC warnings.

---

*Last Updated: 2025-11-30*
