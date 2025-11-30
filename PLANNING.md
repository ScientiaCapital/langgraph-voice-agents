# PLANNING.md - Architecture Decisions & Design Patterns

**Project:** LangGraph Voice-Enabled Agent Framework
**Last Updated:** 2025-11-30

---

## Critical Architectural Rules

### ❌ NEVER
- Use OpenAI for anything (voice, LLM, embeddings)
- Hardcode API keys in code
- Create agents without VoiceAgent base class
- Skip LLM router failover logic

### ✅ ALWAYS
- Use Cartesia for ALL voice (TTS: sonic-2, STT: ink-whisper)
- Use LLMRouter for multi-provider support
- Put API keys in `.env` only
- Test with `main.py --test` before merging

---

## 1. Module Structure

### Directory Layout
```
/Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/
├── voice/                  # Voice I/O (Cartesia + LiveKit)
│   ├── audio_utils.py      # PCM/WAV conversion, RMS amplitude
│   ├── cartesia_client.py  # TTS (sonic-2), STT (ink-whisper)
│   ├── livekit_client.py   # WebRTC rooms (optional)
│   └── __init__.py
│
├── llm/                    # Multi-LLM support (NO OpenAI)
│   ├── provider.py         # Claude, Gemini, OpenRouter clients
│   ├── router.py           # Task-based provider selection + failover
│   └── __init__.py
│
├── core/                   # Base framework
│   ├── base_graph.py       # BaseAgent, AgentState, Mixins
│   ├── state_management.py # SQLite/Redis hybrid persistence
│   └── __init__.py
│
├── agents/                 # Specialized voice agents
│   ├── voice_agent.py      # Base class for all voice agents
│   ├── general_assistant.py # Atlas - General Q&A, conversation
│   ├── code_assistant.py    # Cipher - Code generation, review
│   ├── task_manager.py      # Taskmaster - Task tracking, planning
│   └── __init__.py
│
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies (NO openai package)
├── .env.example            # Environment template
└── CLAUDE.md               # Development log
```

### Design Rationale

**Separation of Concerns:**
- `voice/` - Handles all audio I/O, isolated from business logic
- `llm/` - Manages LLM provider selection, failover, retries
- `core/` - Framework primitives (agent lifecycle, state, mixins)
- `agents/` - Application-level specializations

**Testability:**
- Each module can be tested independently
- Agents can run in text-only mode (no voice dependencies)
- LLM providers can be mocked or swapped

**Extensibility:**
- New agents extend `VoiceAgent` base class
- New LLM providers add to `llm/provider.py`
- New voice providers swap `voice/` module

---

## 2. Specialized Agents

### Agent Hierarchy
```
BaseAgent (core/base_graph.py)
  ↓ Inherits
MultiModalMixin + LLMIntegrationMixin + ErrorHandlingMixin
  ↓ Combines into
VoiceAgent (agents/voice_agent.py)
  ↓ Specializes into
  ├── Atlas (general_assistant.py) - General conversation
  ├── Cipher (code_assistant.py)  - Code assistance
  └── Taskmaster (task_manager.py) - Task management
```

### Agent Comparison Table

| Agent | Class | Specialization | Default LLM | Task Type | Voice ID |
|-------|-------|----------------|-------------|-----------|----------|
| **Atlas** | `GeneralAssistant` | Conversation, Q&A, creative writing, general help | Claude | GENERAL | Default |
| **Cipher** | `CodeAssistant` | Code explanation, review, generation, debugging | Claude | CODING | Default |
| **Taskmaster** | `TaskManager` | Task tracking, project planning, todo management | Gemini | FAST | Default |

### When to Create New Agent

**Create New Agent When:**
- Distinct specialization needed (e.g., medical advisor, legal research)
- Different LLM task type optimal (e.g., image analysis needs vision model)
- Custom voice personality required (different Cartesia voice ID)
- Specialized state management (e.g., patient records, legal docs)

**Extend Existing Agent When:**
- Feature fits existing specialization (e.g., add code execution to Cipher)
- Shares same LLM task type
- No custom voice personality needed
- Uses shared conversation history

---

## 3. LLM Provider Strategy

### Provider Rankings by Task Type

| Task Type | 1st Choice | 2nd Choice | 3rd Choice | Rationale |
|-----------|------------|------------|------------|-----------|
| **GENERAL** | Claude | Gemini | OpenRouter | Claude excels at general conversation, nuanced responses |
| **CODING** | Claude | OpenRouter | Gemini | Claude best for code explanation; OpenRouter has DeepSeek-Coder |
| **REASONING** | Claude | OpenRouter | Gemini | Claude's chain-of-thought reasoning superior |
| **FAST** | Gemini | OpenRouter | Claude | Gemini Flash optimized for low latency |
| **CREATIVE** | Claude | Gemini | OpenRouter | Claude produces more creative, natural writing |

### Automatic Failover Logic

```python
# LLMRouter automatically tries providers in rank order
response = await router.generate(
    messages=[{"role": "user", "content": "Hello"}],
    task_type=TaskType.GENERAL  # Tries: Claude → Gemini → OpenRouter
)

# Fallback cascade:
# 1. Try primary provider (Claude)
# 2. On failure, try secondary (Gemini)
# 3. On failure, try tertiary (OpenRouter)
# 4. On all failures, raise RuntimeError
```

### Provider-Specific Models

| Provider | Models Used | API Endpoint |
|----------|-------------|--------------|
| **Claude** | `claude-3-5-sonnet-20241022` | Anthropic API |
| **Gemini** | `gemini-1.5-flash` (fast), `gemini-1.5-pro` (quality) | Google Generative AI |
| **OpenRouter** | `deepseek/deepseek-chat`, `qwen/qwen-2.5-72b-instruct` | OpenRouter API |

### When to Override Provider

**Override Default Provider When:**
- Specific model capabilities needed (e.g., vision, function calling)
- Cost optimization (Gemini Flash cheaper than Claude)
- Rate limit hit on primary provider
- Testing specific provider behavior

**Example:**
```python
# Force Gemini for fast response
response = await agent.generate_response(
    user_message="Quick fact check",
    provider=LLMProvider.GEMINI,  # Override default Claude
    task_type=TaskType.FAST
)
```

---

## 4. Voice Architecture

### Cartesia Integration (NO OpenAI)

**Critical Decision:** Use Cartesia for ALL voice operations, never OpenAI Whisper/TTS.

**Rationale:**
- Cartesia provides both TTS and STT in one API
- Lower latency than OpenAI Whisper
- Better real-time streaming support
- Consistent voice quality
- Aligns with "NO OpenAI" project rule

### Voice Models

| Feature | Model | Sample Rate | Format | Latency |
|---------|-------|-------------|--------|---------|
| **TTS** | `sonic-2` | 22050 Hz | PCM s16le | <500ms |
| **STT** | `ink-whisper` | 16000 Hz | PCM s16le | <300ms |

### Voice Processing Pipeline

```
User speaks → Microphone
  ↓
Audio Capture (PyAudio)
  ↓
PCM s16le @ 16000 Hz
  ↓
Cartesia STT (ink-whisper)
  ↓
Text transcription
  ↓
LLMRouter (Claude/Gemini/OpenRouter)
  ↓
Text response
  ↓
Cartesia TTS (sonic-2)
  ↓
PCM s16le @ 22050 Hz
  ↓
Audio Playback (PyAudio)
  ↓
User hears response
```

### Voice Activity Detection (VAD)

**Parameters:**
- **RMS Threshold:** Auto-calibrated based on ambient noise
- **Silence Timeout:** 1.5s (stop listening after silence)
- **Min Speech Duration:** 0.3s (filter out noise bursts)

**Algorithm:**
```python
def detect_speech(audio_chunk: bytes) -> bool:
    """RMS amplitude-based speech detection"""
    rms = calculate_rms(audio_chunk)
    return rms > ambient_noise_floor * sensitivity_multiplier
```

### LiveKit Integration (Optional)

**When to Use LiveKit:**
- Multi-user voice sessions needed
- WebRTC real-time requirements
- Participant management required
- Broadcasting to multiple listeners

**When NOT to Use LiveKit:**
- Single-user conversations (use Cartesia directly)
- Text-only mode
- No WebRTC infrastructure available

**LiveKit + Cartesia Architecture:**
```
User A ←→ LiveKit Room ←→ User B
           ↓
    CartesiaTTS/STT
           ↓
    LangGraph Agent
           ↓
      LLMRouter
```

---

## 5. State Management

### Hybrid Storage Strategy

| Data Type | Storage | Rationale |
|-----------|---------|-----------|
| **Conversation History** | In-memory + SQLite | Fast access, persistent across sessions |
| **Agent State** | Redis (optional) | Distributed sessions, real-time updates |
| **User Preferences** | SQLite | Long-term storage, relational queries |
| **Session Cache** | Redis | Ephemeral, high-speed lookups |

### Conversation History Design

**Pattern:**
```python
@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    voice_used: bool

# Stored in VoiceAgent._conversation_history
# Max 20 turns (configurable via VoiceAgentConfig.max_history_turns)
# Trimmed FIFO when limit exceeded
```

**Rationale:**
- Keep context window manageable for LLMs
- Preserve recent context for coherent conversations
- Track voice vs. text mode per turn
- Enable conversation export/replay

### SQLite Schema

```sql
CREATE TABLE conversation_sessions (
    session_id TEXT PRIMARY KEY,
    agent_name TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE conversation_turns (
    turn_id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,  -- 'user' or 'assistant'
    content TEXT,
    timestamp TIMESTAMP,
    voice_used BOOLEAN,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
);

CREATE TABLE user_preferences (
    user_id TEXT PRIMARY KEY,
    preferred_voice_id TEXT,
    preferred_llm_provider TEXT,
    vad_sensitivity REAL,
    created_at TIMESTAMP
);
```

---

## 6. Error Handling & Resilience

### LLM Failover Strategy

**Levels of Fallback:**
1. **Provider Fallback** - Try secondary/tertiary providers
2. **Retry with Backoff** - Exponential backoff for transient errors
3. **Graceful Degradation** - Return cached/default response
4. **User Notification** - Inform user of temporary unavailability

**Example:**
```python
async def generate_with_retry(self, query: str, max_retries: int = 3):
    """LLM generation with automatic retry"""
    for attempt in range(max_retries):
        try:
            return await self.llm_router.chat(query)
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                logger.error(f"All retries exhausted: {e}")
                return "I'm experiencing technical difficulties. Please try again."
```

### Voice Pipeline Errors

| Error Type | Handling | User Experience |
|------------|----------|-----------------|
| **Microphone unavailable** | Fallback to text mode | "Voice input unavailable, using text" |
| **Cartesia API down** | Disable voice, use text | "Voice temporarily disabled" |
| **STT timeout** | Prompt user to repeat | "Sorry, I didn't catch that. Please repeat?" |
| **TTS failure** | Show text response | Display text, skip audio |

---

## 7. Configuration Management

### Environment Variables

**Required (at least one LLM):**
```bash
ANTHROPIC_API_KEY=sk-ant-...  # Claude
GOOGLE_API_KEY=AIza...         # Gemini
OPENROUTER_API_KEY=sk-or-...   # OpenRouter
```

**Voice (required for voice mode):**
```bash
CARTESIA_API_KEY=...
```

**LiveKit (optional):**
```bash
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

**Other:**
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
REDIS_URL=redis://localhost:6379/0  # Optional
```

### Configuration Hierarchy

```
Environment Variables (.env)
  ↓ Override
VoiceAgentConfig (per-agent settings)
  ↓ Override
Runtime Parameters (method arguments)
```

**Example:**
```python
# .env: ANTHROPIC_API_KEY=...
# VoiceAgentConfig: default_provider=LLMProvider.CLAUDE
# Runtime: provider=LLMProvider.GEMINI (overrides all)

response = await agent.generate_response(
    query,
    provider=LLMProvider.GEMINI  # Highest priority
)
```

---

## 8. Testing Strategy

### Test Pyramid

```
           ┌──────────────────┐
           │  E2E Tests       │  ← Manual testing via main.py
           │  (Interactive)   │
           ├──────────────────┤
           │ Integration Tests│  ← main.py --test
           │  (main.py)       │
           ├──────────────────────┤
           │   Unit Tests          │  ← pytest tests/
           │   (per module)        │
           └──────────────────────┘
```

### Test Coverage

| Layer | Test Type | Location | Command |
|-------|-----------|----------|---------|
| **Unit** | Module-level | `tests/test_*.py` | `pytest tests/` |
| **Integration** | Cross-module | `main.py --test` | `python main.py --test` |
| **E2E** | User interaction | Interactive CLI | `python main.py --agent general` |

### Critical Test Cases

**LLM Router:**
- [ ] Provider availability detection
- [ ] Automatic failover on error
- [ ] Task-based provider selection
- [ ] Streaming responses

**Voice Pipeline:**
- [ ] Cartesia TTS/STT initialization
- [ ] Audio format conversion (PCM s16le)
- [ ] VAD speech detection
- [ ] Voice mode toggle (on/off)

**Agents:**
- [ ] Agent creation (all 3 specializations)
- [ ] Text mode responses
- [ ] Voice mode responses (if Cartesia configured)
- [ ] Conversation history management
- [ ] Graceful shutdown

---

## 9. Performance Considerations

### Latency Targets

| Operation | Target | Acceptable | Rationale |
|-----------|--------|------------|-----------|
| **LLM Response** | <2s | <5s | User expects conversational pace |
| **TTS Synthesis** | <500ms | <1s | Real-time voice quality |
| **STT Transcription** | <300ms | <800ms | Minimal lag in conversation |
| **Voice Mode Toggle** | <100ms | <300ms | Instant UI feedback |

### Optimization Strategies

**LLM:**
- Use Gemini Flash for FAST task type (lower latency than Claude)
- Cache frequent responses in Redis
- Stream responses for perceived speed

**Voice:**
- Pre-buffer TTS audio chunks
- Asynchronous STT processing
- Parallel LLM + TTS pipeline

**State:**
- In-memory conversation history (avoid DB reads per message)
- Lazy-load SQLite (only on session restore)
- Redis for session state (faster than SQLite)

---

## 10. Future Architecture Decisions

### Planned Enhancements

1. **Multi-Modal Input**
   - Vision models (Gemini 1.5 Pro with images)
   - Document parsing (PDF, DOCX)
   - Screen capture analysis

2. **Extended LLM Providers**
   - Local models (Ollama, LM Studio)
   - Cerebras (ultra-fast inference)
   - Groq (optimized hardware)

3. **Advanced Voice Features**
   - Voice cloning (custom voice IDs)
   - Emotion detection from voice
   - Multi-language STT/TTS

4. **Distributed Architecture**
   - Kubernetes deployment
   - Load balancing across agents
   - Shared state via Redis cluster

### Open Questions

1. **Should we support OpenAI as LLM provider?**
   - **Current Decision:** NO - violates project rule
   - **Revisit:** If user explicitly requests, make optional

2. **LiveKit vs. direct WebRTC?**
   - **Current Decision:** LiveKit for WebRTC (abstraction layer)
   - **Alternative:** Roll own WebRTC with aiortc

3. **SQLite vs. PostgreSQL for state?**
   - **Current Decision:** SQLite (simpler, embedded)
   - **Consider PostgreSQL:** If multi-user, distributed deployment

---

## 11. Architecture Evolution Log

| Date | Decision | Rationale | Files Affected |
|------|----------|-----------|----------------|
| 2025-01-28 | Remove all OpenAI dependencies | Project rule: NO OpenAI | All modules |
| 2025-01-28 | Use Cartesia for voice | Unified TTS/STT, lower latency | `voice/` module |
| 2025-01-28 | Multi-LLM with router | Provider diversity, resilience | `llm/router.py` |
| 2025-01-28 | Three specialized agents | Domain expertise vs. generic | `agents/` |
| 2025-01-28 | VoiceAgent base class | Code reuse, consistent patterns | `agents/voice_agent.py` |

---

*Last Updated: 2025-11-30*
*Architecture Version: 2.0 (Post-OpenAI Removal)*
