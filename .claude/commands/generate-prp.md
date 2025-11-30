---
description: "Generate Project Requirements Plan (PRP) for langgraph-voice-agents features"
---

# Generate PRP for langgraph-voice-agents

**Critical Rules:**
- **NO OpenAI** - Use Claude, Gemini, OpenRouter only
- **Cartesia for all voice** - TTS (sonic-2) and STT (ink-whisper)
- API keys in `.env` only

---

## PRP Generation Process

You are creating a detailed implementation plan for adding a new feature to the **langgraph-voice-agents** project.

### Step 1: Understanding Current Architecture

**Project Structure:**
```
/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/
├── voice/              # Cartesia TTS/STT, LiveKit WebRTC
├── llm/                # Claude, Gemini, OpenRouter routing
├── core/               # BaseAgent, state management
├── agents/             # VoiceAgent base + 3 specialized agents
└── main.py             # CLI entry point
```

**Three Specialized Agents:**
1. **Atlas** (`general_assistant.py`) - General conversation, Q&A, creative writing
2. **Cipher** (`code_assistant.py`) - Code explanation, review, generation
3. **Taskmaster** (`task_manager.py`) - Task tracking, project planning

**Key Design Patterns:**
- VoiceAgent base class with MultiModal + LLMIntegration mixins
- LLMRouter with automatic failover (Claude → Gemini → OpenRouter)
- Cartesia WebSocket for TTS/STT (NO OpenAI Whisper)
- SQLite/Redis hybrid state management

---

### Step 2: Analyze Feature Request

Ask the user clarifying questions:

1. **Which agent(s) does this feature affect?**
   - Atlas (general)
   - Cipher (code)
   - Taskmaster (tasks)
   - New agent?
   - All agents (VoiceAgent base class)?

2. **Does this feature require voice capabilities?**
   - Voice input (Cartesia STT)?
   - Voice output (Cartesia TTS)?
   - Text-only?

3. **Which LLM provider(s) should be used?**
   - Claude (best for general/reasoning/creative)
   - Gemini (best for fast responses)
   - OpenRouter (DeepSeek, Qwen for coding)
   - Multiple with fallback?

4. **Does this require new state management?**
   - Conversation history extension?
   - New SQLite tables?
   - Redis caching?

5. **LiveKit integration needed?**
   - WebRTC multi-user rooms?
   - Real-time audio streaming?

---

### Step 3: Create PRP Template

Use the base template from `/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/PRPs/templates/prp_base.md`

Fill in these sections:

#### 1. Feature Overview
- **Name**: Clear, descriptive name
- **Target Agent(s)**: Which agent(s) this affects
- **Complexity**: Low/Medium/High
- **Estimated Effort**: Hours or days

#### 2. Technical Requirements

**LLM Requirements:**
- Primary provider: Claude/Gemini/OpenRouter?
- Task type: GENERAL/CODING/REASONING/FAST/CREATIVE
- Fallback strategy: Automatic or manual?
- Example:
  ```python
  # Use Gemini for fast responses
  response = await self.generate_response(
      user_message=input_data,
      task_type=TaskType.FAST,
      provider=LLMProvider.GEMINI
  )
  ```

**Voice Requirements:**
- TTS needed? Cartesia `sonic-2` model
- STT needed? Cartesia `ink-whisper` model
- Audio processing: PCM s16le format
- Example:
  ```python
  # Generate voice output
  await self.speak("Hello, I'm ready to assist!")
  ```

**State Management:**
- Conversation history: Use existing `_conversation_history`?
- Persistent storage: SQLite or Redis?
- Data models: New Pydantic classes?

#### 3. Implementation Plan

**Phase 1: Setup**
- Create/modify Python files
- Add dependencies to `requirements.txt`
- Update `.env.example` with new keys

**Phase 2: Core Logic**
- Implement main feature logic
- Use VoiceAgent patterns (see template)
- Integrate with LLMRouter

**Phase 3: Voice Integration** (if applicable)
- Add TTS output points
- Add STT input handling
- Configure Cartesia models

**Phase 4: Testing**
- Unit tests in `/tests/`
- Integration test in `main.py --test`
- Manual interactive testing

**Phase 5: Documentation**
- Update `CLAUDE.md`
- Add usage examples
- Document new environment variables

#### 4. Code Examples

Provide concrete code snippets using existing patterns:

**VoiceAgent Pattern:**
```python
from agents import VoiceAgent, VoiceAgentConfig
from llm import LLMProvider, TaskType

class NewAgent(VoiceAgent):
    def __init__(self, voice_mode: bool = True):
        config = VoiceAgentConfig(
            agent_name="Agent Name",
            agent_description="What it does",
            task_type=TaskType.GENERAL,  # or CODING, FAST, etc.
            default_provider=LLMProvider.CLAUDE,
            greeting_message="Hello! How can I help?"
        )
        super().__init__(config, mode=AgentMode.VOICE if voice_mode else AgentMode.TEXT)
```

**LLM Router with Fallback:**
```python
from llm import LLMRouter, TaskType, LLMProvider

async def intelligent_response(self, query: str):
    """Use LLM router with task-based provider selection"""
    try:
        # Router automatically selects best provider for task
        response = await self.llm_router.chat(
            user_message=query,
            task_type=TaskType.CODING,  # Prefer Claude, fallback to OpenRouter
            conversation_history=self._build_history_for_llm()
        )
        return response
    except Exception as e:
        logger.error(f"All LLM providers failed: {e}")
        return "I'm having trouble connecting. Please try again."
```

**Cartesia Voice Pattern:**
```python
async def handle_voice_interaction(self, audio_data: bytes):
    """Process voice input and respond with voice output"""
    # Transcribe with Cartesia STT (ink-whisper)
    text = await self.listen(audio_data)

    # Process with LLM
    response = await self.process_input(text)

    # Synthesize with Cartesia TTS (sonic-2)
    await self.speak(response)
```

#### 5. File Changes

List all files that need to be created or modified:

```
CREATE:
  /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/agents/new_agent.py

MODIFY:
  /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/agents/__init__.py
  /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/main.py
  /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/requirements.txt (if new deps)
  /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/CLAUDE.md

UPDATE DOCS:
  /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/README.md
```

#### 6. Validation Steps

How to verify the feature works:

```bash
# Step 1: Run validation
python /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/main.py --test

# Step 2: Start new agent
python /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/main.py --agent new

# Step 3: Test voice mode
# In interactive session:
# > voice on
# > [test feature]
# > voice off

# Step 4: Verify no OpenAI dependencies
grep -ri "openai" /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents --exclude-dir=.git
```

---

### Step 4: Save PRP

Save the completed PRP to:
```
/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/PRPs/YYYY-MM-DD_feature_name.md
```

---

## PRP Template Variables

When generating, fill in:

- `{{FEATURE_NAME}}` - Clear, descriptive name
- `{{TARGET_AGENTS}}` - Atlas/Cipher/Taskmaster/VoiceAgent/New
- `{{LLM_PROVIDER}}` - Claude/Gemini/OpenRouter
- `{{TASK_TYPE}}` - GENERAL/CODING/REASONING/FAST/CREATIVE
- `{{VOICE_REQUIRED}}` - Yes/No
- `{{NEW_DEPENDENCIES}}` - List new pip packages
- `{{ENV_VARS}}` - List new .env variables

---

## Example PRPs

### Example 1: Add Streaming Response to General Agent

**Feature:** Real-time streaming LLM responses with live voice synthesis

**Target Agent:** Atlas (general_assistant.py)

**LLM:** Claude with streaming API

**Voice:** Cartesia TTS chunks as tokens arrive

### Example 2: Add Code Execution to Cipher

**Feature:** Execute Python code snippets safely in sandbox

**Target Agent:** Cipher (code_assistant.py)

**LLM:** OpenRouter (DeepSeek-Coder) for code generation

**Voice:** Read execution results via Cartesia TTS

### Example 3: Add Multi-User Voice Rooms

**Feature:** Multiple users in LiveKit room with Taskmaster coordination

**Target Agent:** Taskmaster (task_manager.py)

**LLM:** Gemini (fast responses for real-time)

**Voice:** LiveKit WebRTC + Cartesia for all participants

---

## Critical Reminders

### ❌ DO NOT
- Use OpenAI for anything (voice, LLM, embeddings)
- Hardcode API keys in any file
- Skip LLM router failover logic
- Mix OpenAI Whisper/TTS with Cartesia

### ✅ DO
- Use Cartesia for ALL voice (TTS: sonic-2, STT: ink-whisper)
- Use LLMRouter for multi-provider support
- Put ALL API keys in `.env` only
- Test with `main.py --test` before merging
- Update `CLAUDE.md` with architecture changes

---

*Last Updated: 2025-11-30*
