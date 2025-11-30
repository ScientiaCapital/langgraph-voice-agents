# PRP: {{FEATURE_NAME}}

**Date:** {{DATE}}
**Target Agent(s):** {{TARGET_AGENTS}}
**Complexity:** {{COMPLEXITY}}
**Estimated Effort:** {{EFFORT}}

---

## Critical Rules

- **NO OpenAI** - Use Claude, Gemini, OpenRouter only
- **Cartesia for all voice** - TTS (sonic-2) and STT (ink-whisper)
- API keys in `.env` only, never hardcoded

---

## 1. Feature Overview

### Description
{{FEATURE_DESCRIPTION}}

### Motivation
{{WHY_THIS_FEATURE}}

### Success Criteria
- [ ] {{CRITERION_1}}
- [ ] {{CRITERION_2}}
- [ ] {{CRITERION_3}}

---

## 2. Technical Requirements

### LLM Requirements

**Primary Provider:** {{LLM_PROVIDER}}
**Task Type:** `TaskType.{{TASK_TYPE}}`
**Fallback Strategy:** {{FALLBACK_STRATEGY}}

**Provider Selection Rationale:**
- {{PROVIDER_CHOICE_REASON}}

**LLM Router Pattern:**
```python
# Use LLM router with automatic failover
from llm import LLMRouter, TaskType, LLMProvider

async def process_query(self, query: str) -> str:
    """
    Process query with optimal LLM provider.
    Router automatically selects best provider based on task type.
    Falls back to secondary providers if primary fails.
    """
    response = await self.generate_response(
        user_message=query,
        task_type=TaskType.{{TASK_TYPE}},  # GENERAL, CODING, REASONING, FAST, CREATIVE
        provider=LLMProvider.{{PRIMARY_PROVIDER}},  # Optional: Claude, Gemini, OpenRouter
        conversation_history=self._build_history_for_llm(),
    )
    return response
```

**Provider Rankings by Task:**
```
GENERAL:   Claude → Gemini → OpenRouter
CODING:    Claude → OpenRouter → Gemini
REASONING: Claude → OpenRouter → Gemini
FAST:      Gemini → OpenRouter → Claude
CREATIVE:  Claude → Gemini → OpenRouter
```

---

### Voice Requirements

**Voice Needed:** {{YES_OR_NO}}

**If YES:**

**TTS (Text-to-Speech):**
- Model: Cartesia `sonic-2`
- Sample Rate: 22050 Hz
- Format: PCM s16le

**STT (Speech-to-Text):**
- Model: Cartesia `ink-whisper`
- Sample Rate: 16000 Hz
- Format: PCM s16le

**Voice Integration Pattern:**
```python
from agents import VoiceAgent, VoiceAgentConfig
from core import AgentMode

class {{AGENT_CLASS_NAME}}(VoiceAgent):
    """
    {{AGENT_DESCRIPTION}}

    Voice: Cartesia (sonic-2 TTS, ink-whisper STT)
    LLM: {{LLM_PROVIDER}}
    NO OpenAI dependencies.
    """

    def __init__(self, voice_mode: bool = True):
        config = VoiceAgentConfig(
            agent_name="{{AGENT_DISPLAY_NAME}}",
            agent_description="{{AGENT_DESCRIPTION}}",
            task_type=TaskType.{{TASK_TYPE}},
            default_provider=LLMProvider.{{PRIMARY_PROVIDER}},
            greeting_message="{{GREETING}}",
        )

        super().__init__(
            config=config,
            mode=AgentMode.VOICE if voice_mode else AgentMode.TEXT
        )

    async def handle_voice_interaction(self, audio_data: bytes):
        """Process voice input and respond with voice"""
        # STT: Cartesia ink-whisper
        transcription = await self.listen(audio_data)

        if not transcription.strip():
            return

        # Process with LLM
        response = await self.process_input(transcription)

        # TTS: Cartesia sonic-2 (auto-called if voice mode)
        # No explicit speak() needed - VoiceAgent handles it
```

**Voice Activity Detection (VAD):**
- RMS amplitude threshold: {{VAD_THRESHOLD}} (default: auto)
- Silence timeout: {{SILENCE_TIMEOUT}}s (default: 1.5s)
- Min speech duration: {{MIN_SPEECH}}s (default: 0.3s)

---

### State Management

**Conversation History:** {{USE_EXISTING_OR_NEW}}

**If NEW:**
- Storage: {{SQLITE_OR_REDIS}}
- Data Model:
  ```python
  from dataclasses import dataclass
  from datetime import datetime

  @dataclass
  class {{STATE_CLASS_NAME}}:
      {{FIELD_1}}: {{TYPE_1}}
      {{FIELD_2}}: {{TYPE_2}}
      timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
  ```

**Persistent Storage Pattern:**
```python
from core.state_management import StateManager

class {{AGENT_CLASS_NAME}}(VoiceAgent):
    def __init__(self, voice_mode: bool = True):
        super().__init__(...)

        # Initialize state manager
        self.state_manager = StateManager(
            db_path="{{DB_PATH}}",
            redis_url="{{REDIS_URL}}"  # Optional
        )

    async def save_state(self, key: str, value: dict):
        """Persist state to SQLite/Redis"""
        await self.state_manager.set(key, value)

    async def load_state(self, key: str) -> dict:
        """Load state from storage"""
        return await self.state_manager.get(key)
```

---

### LiveKit Integration (Optional)

**Needed:** {{YES_OR_NO}}

**If YES:**

**Use Case:** {{LIVEKIT_USE_CASE}}

**Configuration:**
```python
from voice import LiveKitConfig, LiveKitCartesiaClient

async def start_webrtc_session(self, room_name: str):
    """Start LiveKit WebRTC room with Cartesia voice"""
    livekit_config = LiveKitConfig(
        room_name=room_name,
        participant_name=self.config.agent_name,
    )

    self._livekit_client = LiveKitCartesiaClient(livekit_config)
    self._livekit_client.on_transcription = self._handle_transcription

    connected = await self._livekit_client.connect()
    if not connected:
        logger.error("Failed to connect to LiveKit")
        return False

    logger.info(f"Connected to room: {room_name}")
    return True
```

**Environment Variables:**
```bash
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

---

## 3. Implementation Plan

### Phase 1: Setup
**Goal:** Prepare dependencies and file structure

**Tasks:**
- [ ] Add dependencies to `requirements.txt`:
  ```bash
  {{NEW_PACKAGE_1}}>=1.0.0  # For {{REASON_1}}
  {{NEW_PACKAGE_2}}>=2.0.0  # For {{REASON_2}}
  ```
- [ ] Add environment variables to `.env.example`:
  ```bash
  {{NEW_ENV_VAR_1}}=value  # Purpose: {{PURPOSE_1}}
  ```
- [ ] Create new files:
  ```
  {{FILE_1}}
  {{FILE_2}}
  ```

**Validation:**
```bash
# Test imports
python -c "import {{NEW_MODULE}}; print('✅ Dependencies OK')"
```

---

### Phase 2: Core Implementation
**Goal:** Build main feature logic

**Tasks:**
- [ ] Create `{{PRIMARY_FILE}}` with {{MAIN_FUNCTIONALITY}}
- [ ] Implement LLM integration using router pattern
- [ ] Add error handling and logging
- [ ] Implement core business logic

**Code Structure:**
```python
# File: {{PRIMARY_FILE}}

from agents import VoiceAgent, VoiceAgentConfig
from llm import LLMProvider, TaskType
from core import AgentMode
import logging

logger = logging.getLogger(__name__)


class {{CLASS_NAME}}(VoiceAgent):
    """
    {{CLASS_DESCRIPTION}}

    NO OpenAI - Uses:
    - Voice: Cartesia (sonic-2 TTS, ink-whisper STT)
    - LLM: {{LLM_PROVIDER}} with automatic failover
    """

    def __init__(self, voice_mode: bool = True):
        config = VoiceAgentConfig(
            agent_name="{{AGENT_NAME}}",
            agent_description="{{AGENT_DESCRIPTION}}",
            task_type=TaskType.{{TASK_TYPE}},
            default_provider=LLMProvider.{{PRIMARY_PROVIDER}},
        )

        super().__init__(config, mode=AgentMode.VOICE if voice_mode else AgentMode.TEXT)

    async def {{METHOD_NAME}}(self, {{PARAMS}}) -> {{RETURN_TYPE}}:
        """{{METHOD_DESCRIPTION}}"""
        # Implementation here
        pass
```

**Validation:**
```bash
# Test class instantiation
python -c "
from {{MODULE}} import {{CLASS_NAME}}
import asyncio

async def test():
    agent = {{CLASS_NAME}}(voice_mode=False)
    print(f'✅ Created: {agent.config.agent_name}')
    await agent.close()

asyncio.run(test())
"
```

---

### Phase 3: Voice Integration (if applicable)
**Goal:** Wire up Cartesia TTS/STT

**Tasks:**
- [ ] Add voice input handling (STT: ink-whisper)
- [ ] Add voice output synthesis (TTS: sonic-2)
- [ ] Configure VAD parameters
- [ ] Test audio pipeline

**Voice Pipeline:**
```python
async def handle_voice_input(self, audio_data: bytes) -> str:
    """
    Process voice input through Cartesia STT.

    Args:
        audio_data: PCM s16le audio bytes (16000 Hz for STT)

    Returns:
        Transcribed text
    """
    self._set_voice_state(VoiceAgentState.LISTENING)

    try:
        # Cartesia ink-whisper transcription
        text = await self.listen(audio_data)
        logger.info(f"Transcribed: {text}")
        return text
    except Exception as e:
        logger.error(f"STT failed: {e}")
        return ""
    finally:
        self._set_voice_state(VoiceAgentState.IDLE)


async def respond_with_voice(self, text: str) -> bool:
    """
    Synthesize and speak text using Cartesia TTS.

    Args:
        text: Text to speak

    Returns:
        True if successful
    """
    self._set_voice_state(VoiceAgentState.SPEAKING)

    try:
        # Cartesia sonic-2 synthesis
        success = await self.speak(text)
        logger.info(f"Spoke: {text[:50]}...")
        return success
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return False
    finally:
        self._set_voice_state(VoiceAgentState.IDLE)
```

**Validation:**
```bash
# Test voice mode (requires CARTESIA_API_KEY)
python -c "
import asyncio
import os
from {{MODULE}} import {{CLASS_NAME}}

async def test():
    if not os.getenv('CARTESIA_API_KEY'):
        print('⚠️  CARTESIA_API_KEY not set, skipping')
        return

    agent = {{CLASS_NAME}}(voice_mode=True)
    success = await agent.speak('Testing Cartesia voice')
    print(f'Voice test: {'✅' if success else '❌'}')
    await agent.close()

asyncio.run(test())
"
```

---

### Phase 4: System Integration
**Goal:** Connect to main CLI and existing agents

**Tasks:**
- [ ] Update `agents/__init__.py`:
  ```python
  from agents.{{MODULE}} import {{CLASS_NAME}}

  def create_{{AGENT_FUNCTION_NAME}}(voice_mode: bool = True) -> {{CLASS_NAME}}:
      """Factory function for {{CLASS_NAME}}"""
      return {{CLASS_NAME}}(voice_mode=voice_mode)

  __all__ = [
      # ... existing ...
      "{{CLASS_NAME}}",
      "create_{{AGENT_FUNCTION_NAME}}",
  ]
  ```

- [ ] Update `main.py` agent selection:
  ```python
  # In select_agent():
  print("{{N}}. {{AGENT_DISPLAY_NAME}} ({{SHORT_NAME}})")
  print("   - {{DESCRIPTION}}")

  # In choice handling:
  elif choice == '{{N}}' or choice == '{{SHORT_NAME}}':
      return '{{SHORT_NAME}}'

  # In agent creation:
  if agent_type == '{{SHORT_NAME}}':
      agent = create_{{AGENT_FUNCTION_NAME}}(voice_mode=voice_mode)
  ```

- [ ] Add integration test to `main.py`:
  ```python
  # In run_tests():
  print(f"\n{{N}}. Testing {{AGENT_NAME}} creation...")
  try:
      agent = create_{{AGENT_FUNCTION_NAME}}(voice_mode=False)
      print(f"   ✅ Created: {agent.config.agent_name}")
      tests_passed += 1
      await agent.close()
  except Exception as e:
      print(f"   ❌ {{AGENT_NAME}} failed: {e}")
      tests_failed += 1
  ```

**Validation:**
```bash
# Run integration tests
python main.py --test

# Test agent selection
python main.py --agent {{SHORT_NAME}} --text
```

---

### Phase 5: Documentation
**Goal:** Update all project docs

**Tasks:**
- [ ] Update `CLAUDE.md`:
  ```markdown
  ## Agents (`/agents/`)

  | Agent | Name | Specialization | Default LLM |
  |-------|------|----------------|-------------|
  | ... existing ...
  | `{{MODULE}}.py` | {{AGENT_NAME}} | {{SPECIALIZATION}} | {{LLM_PROVIDER}} |

  ## Session History

  | Session | Date | Summary |
  |---------|------|---------|
  | ... existing ...
  | {{SESSION_N}} | {{DATE}} | Added {{AGENT_NAME}} with {{FEATURES}} |
  ```

- [ ] Update `README.md`:
  ```markdown
  ## {{AGENT_NAME}}

  ```bash
  python main.py --agent {{SHORT_NAME}}
  ```

  **Features:**
  - {{FEATURE_1}}
  - {{FEATURE_2}}

  **Voice:** Cartesia (sonic-2 TTS, ink-whisper STT)
  **LLM:** {{LLM_PROVIDER}} with automatic failover
  ```

- [ ] Add code comments and docstrings

**Validation:**
```bash
# Check markdown formatting
cat README.md CLAUDE.md
```

---

### Phase 6: Final Validation
**Goal:** Comprehensive testing

**Tasks:**
- [ ] Run full test suite: `python main.py --test`
- [ ] Code quality: `black --check . && mypy .`
- [ ] Verify NO OpenAI: `grep -ri "openai" . --exclude-dir=.git`
- [ ] Interactive testing (see checklist below)
- [ ] Git commit changes

**Interactive Test Checklist:**
```bash
python main.py --agent {{SHORT_NAME}}
```

- [ ] Agent starts without errors
- [ ] Responds to test query: "{{TEST_QUERY}}"
- [ ] `voice on` enables voice (if Cartesia configured)
- [ ] `voice off` disables voice
- [ ] `clear` resets history
- [ ] `quit` exits cleanly
- [ ] LLM responses are coherent
- [ ] {{FEATURE_SPECIFIC_TEST}}

**Validation:**
```bash
# All tests should pass
python main.py --test

# No OpenAI dependencies
! grep -ri "openai" . --exclude-dir=.git --exclude-dir=PRPs --exclude="*.md"
```

---

## 4. File Changes

### Files to CREATE
```
{{CREATE_FILE_1}}
{{CREATE_FILE_2}}
```

### Files to MODIFY
```
{{MODIFY_FILE_1}}
{{MODIFY_FILE_2}}
agents/__init__.py
main.py
CLAUDE.md
README.md
```

### Files to UPDATE (if needed)
```
requirements.txt
.env.example
```

---

## 5. Testing Strategy

### Unit Tests
```python
# File: tests/test_{{MODULE}}.py

import pytest
import asyncio
from {{MODULE}} import {{CLASS_NAME}}

@pytest.mark.asyncio
async def test_{{FEATURE}}_creation():
    """Test agent can be created"""
    agent = {{CLASS_NAME}}(voice_mode=False)
    assert agent.config.agent_name == "{{AGENT_NAME}}"
    await agent.close()


@pytest.mark.asyncio
async def test_{{FEATURE}}_response():
    """Test agent responds to input"""
    agent = {{CLASS_NAME}}(voice_mode=False)
    response = await agent.process_input("{{TEST_INPUT}}")
    assert len(response) > 0
    await agent.close()
```

### Integration Tests
```bash
# Run from project root
python main.py --test

# Expected output:
# ✅ LLM Router initialized
# ✅ Created: {{AGENT_NAME}}
# ✅ All tests passed
```

### Manual Testing
```bash
# Text mode
python main.py --agent {{SHORT_NAME}} --text

# Voice mode (requires CARTESIA_API_KEY)
python main.py --agent {{SHORT_NAME}}
```

---

## 6. Rollback Plan

If implementation fails:

```bash
# Stash changes
git stash

# Verify baseline works
python main.py --test

# Review what broke
git stash show -p

# Option 1: Fix and retry
git stash pop
# ... fix issues ...
python main.py --test

# Option 2: Abandon changes
git stash drop
```

---

## 7. Deployment Checklist

Before merging:

- [ ] All tests pass (`python main.py --test`)
- [ ] Code formatted (`black .`)
- [ ] Type checks pass (`mypy . --ignore-missing-imports`)
- [ ] NO OpenAI dependencies (`grep -ri "openai"` returns empty)
- [ ] Cartesia used for ALL voice operations
- [ ] At least one LLM provider works (Claude/Gemini/OpenRouter)
- [ ] Documentation updated (CLAUDE.md, README.md)
- [ ] `.env.example` has all required variables
- [ ] Interactive testing complete

---

## 8. Open Questions

{{QUESTION_1}}
{{QUESTION_2}}

---

## 9. Future Enhancements

{{ENHANCEMENT_1}}
{{ENHANCEMENT_2}}

---

## 10. References

- **VoiceAgent Base Class:** `/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/agents/voice_agent.py`
- **LLM Router:** `/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/llm/router.py`
- **Cartesia Client:** `/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/voice/cartesia_client.py`
- **Main CLI:** `/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/main.py`

---

*Template Version: 1.0*
*Last Updated: 2025-11-30*
