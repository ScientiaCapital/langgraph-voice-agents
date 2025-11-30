# TASK.md - Current Work Tracking

**Project:** LangGraph Voice-Enabled Agent Framework
**Last Updated:** 2025-11-30
**Status:** Session 2 Complete - Clean Foundation Established

---

## Critical Rules (Always Active)

- **NO OpenAI** - Use Claude, Gemini, OpenRouter only
- **Cartesia for all voice** - TTS (sonic-2) and STT (ink-whisper)
- API keys in `.env` only, never hardcoded
- Test with `python main.py --test` before any commit

---

## Current Session Status

### Session 2 (2025-01-28) - ✅ COMPLETE

**Goal:** Complete rebuild removing OpenAI, establishing clean multi-LLM + Cartesia foundation

**Accomplishments:**
- ✅ Removed ALL OpenAI dependencies (Whisper, TTS, API client)
- ✅ Integrated Cartesia for TTS (sonic-2) and STT (ink-whisper)
- ✅ Built multi-LLM router (Claude, Gemini, OpenRouter) with automatic failover
- ✅ Created three specialized agents (Atlas, Cipher, Taskmaster)
- ✅ Implemented VoiceAgent base class with mixins
- ✅ Added LiveKit WebRTC support (optional)
- ✅ Built CLI with agent selection and voice toggle
- ✅ Created comprehensive integration tests

**Files Created/Modified:**
```
voice/audio_utils.py          - PCM/WAV conversion, RMS amplitude
voice/cartesia_client.py      - Cartesia TTS/STT WebSocket client
voice/livekit_client.py       - LiveKit WebRTC integration
llm/provider.py               - Claude, Gemini, OpenRouter clients
llm/router.py                 - Task-based LLM routing with failover
core/base_graph.py            - BaseAgent, mixins
core/state_management.py      - SQLite/Redis hybrid storage
agents/voice_agent.py         - VoiceAgent base class
agents/general_assistant.py   - Atlas (general conversation)
agents/code_assistant.py      - Cipher (code assistance)
agents/task_manager.py        - Taskmaster (task management)
main.py                       - CLI entry point with tests
```

**Removed from Session 1:**
```
❌ tools/ directory (MCP adapters)
❌ agents/task_orchestrator.py
❌ agents/task_executor.py
❌ agents/task_checker.py
❌ All OpenAI imports/references
```

---

## Session 3 (Future) - Planning Phase

### Goals
1. **Extended Agent Capabilities**
   - Streaming LLM responses with live voice synthesis
   - Conversation export/import
   - Multi-turn context summarization

2. **More LLM Integrations**
   - Cerebras (ultra-fast inference)
   - Local models via Ollama
   - Groq (optimized hardware)

3. **Advanced Voice Features**
   - Voice personality selection (custom Cartesia voice IDs)
   - Multi-language support (STT/TTS)
   - Voice emotion detection

4. **Production Readiness**
   - Docker containerization
   - Kubernetes deployment manifests
   - Production logging/monitoring
   - Rate limiting and quotas

---

## Work Queue (Prioritized)

### High Priority (Next Session)

#### 1. Streaming Response with Live Voice
**Status:** Not Started
**Complexity:** Medium
**Effort:** 2-3 hours

**Description:**
Stream LLM responses token-by-token with real-time Cartesia TTS synthesis.

**Tasks:**
- [ ] Implement `stream_with_voice()` in VoiceAgent
- [ ] Buffer tokens into sentences for TTS
- [ ] Play audio chunks as they arrive
- [ ] Handle interruptions mid-stream

**Files:**
- `agents/voice_agent.py` - Add streaming method
- `voice/cartesia_client.py` - Support streaming TTS

**PRP:** Create `PRPs/2025-11-30_streaming_voice_responses.md`

---

#### 2. Conversation Export/Import
**Status:** Not Started
**Complexity:** Low
**Effort:** 1-2 hours

**Description:**
Save conversation history to JSON/CSV, reload on agent restart.

**Tasks:**
- [ ] Add `export_conversation()` method
- [ ] Add `import_conversation()` method
- [ ] Support JSON and CSV formats
- [ ] CLI command: `save`, `load`

**Files:**
- `agents/voice_agent.py` - Export/import methods
- `main.py` - Add CLI commands

**PRP:** Create `PRPs/2025-11-30_conversation_export_import.md`

---

#### 3. Multi-Turn Context Summarization
**Status:** Not Started
**Complexity:** Medium
**Effort:** 2-3 hours

**Description:**
Automatically summarize conversation when history exceeds limit, preserve key context.

**Tasks:**
- [ ] Detect when history exceeds max_history_turns
- [ ] Use LLM to summarize oldest turns
- [ ] Replace old turns with summary
- [ ] Test context preservation

**Files:**
- `agents/voice_agent.py` - Summarization logic
- `llm/router.py` - Add SUMMARIZATION task type

**PRP:** Create `PRPs/2025-11-30_context_summarization.md`

---

### Medium Priority (Future Sessions)

#### 4. Add Cerebras LLM Provider
**Status:** Not Started
**Complexity:** Low
**Effort:** 1-2 hours

**Description:**
Integrate Cerebras for ultra-fast inference (70B model, <1s response time).

**Tasks:**
- [ ] Add `CerebrasClient` to `llm/provider.py`
- [ ] Update `LLMRouter` with Cerebras fallback
- [ ] Add `CEREBRAS_API_KEY` to `.env.example`
- [ ] Test FAST task type with Cerebras

**Files:**
- `llm/provider.py` - New CerebrasClient
- `llm/router.py` - Update rankings for FAST tasks
- `requirements.txt` - Add `cerebras` package

**PRP:** Create `PRPs/YYYY-MM-DD_cerebras_integration.md`

---

#### 5. Local LLM Support (Ollama)
**Status:** Not Started
**Complexity:** Medium
**Effort:** 3-4 hours

**Description:**
Run local LLMs via Ollama for privacy/offline use.

**Tasks:**
- [ ] Add `OllamaClient` to `llm/provider.py`
- [ ] Auto-detect Ollama server (localhost:11434)
- [ ] Support model selection (llama3.2, codellama, etc.)
- [ ] Fallback to cloud if Ollama unavailable

**Files:**
- `llm/provider.py` - New OllamaClient
- `llm/router.py` - Add LOCAL task type
- `requirements.txt` - Add `ollama` package

**PRP:** Create `PRPs/YYYY-MM-DD_ollama_integration.md`

---

#### 6. Voice Personality Selection
**Status:** Not Started
**Complexity:** Low
**Effort:** 1-2 hours

**Description:**
Allow users to choose Cartesia voice ID (personality, accent, gender).

**Tasks:**
- [ ] Add voice ID catalog to `voice/cartesia_client.py`
- [ ] CLI command: `voice list`, `voice set <id>`
- [ ] Save preference in user_preferences table
- [ ] Preview voice samples

**Files:**
- `voice/cartesia_client.py` - Voice catalog
- `main.py` - Voice selection CLI
- `core/state_management.py` - Save preferences

**PRP:** Create `PRPs/YYYY-MM-DD_voice_personality.md`

---

### Low Priority (Backlog)

#### 7. Docker Containerization
**Status:** Not Started
**Complexity:** Low
**Effort:** 2-3 hours

**Tasks:**
- [ ] Create `Dockerfile` for CLI
- [ ] Create `docker-compose.yml` with Redis
- [ ] Document Docker usage in README.md
- [ ] Test in Docker environment

---

#### 8. Kubernetes Deployment
**Status:** Not Started
**Complexity:** High
**Effort:** 1-2 days

**Tasks:**
- [ ] Create K8s manifests (`deployment.yaml`, `service.yaml`)
- [ ] Set up secrets management
- [ ] Configure Redis StatefulSet
- [ ] Load balancing across agent replicas

---

#### 9. Production Logging
**Status:** Not Started
**Complexity:** Medium
**Effort:** 2-3 hours

**Tasks:**
- [ ] Replace print() with structured logging
- [ ] Add log rotation
- [ ] Integration with Sentry/Datadog
- [ ] Request ID tracking

---

#### 10. Rate Limiting & Quotas
**Status:** Not Started
**Complexity:** Medium
**Effort:** 3-4 hours

**Tasks:**
- [ ] Track API calls per user/session
- [ ] Enforce quotas (e.g., 100 messages/hour)
- [ ] Graceful degradation on quota exceeded
- [ ] Dashboard for usage monitoring

---

## Blocked/On Hold

**None currently.**

---

## Completed Work (Archive)

### Session 1 (2025-01-21) - ✅ COMPLETE
- Initial implementation with MCP tools
- OpenAI Whisper + TTS integration
- Task orchestrator agents

**Outcome:** Scrapped in Session 2 due to OpenAI dependency requirement change.

### Session 2 (2025-01-28) - ✅ COMPLETE
See "Current Session Status" above.

---

## Testing Checklist (Before Each Commit)

Run these before any `git commit`:

```bash
# 1. Environment check
python main.py --test

# 2. Code quality
black --check .
mypy . --ignore-missing-imports

# 3. Verify NO OpenAI dependencies
! grep -ri "openai" . --exclude-dir=.git --exclude-dir=PRPs --exclude="*.md"

# 4. Interactive test (manual)
python main.py --agent general --text
# > Test query
# > voice on (if Cartesia configured)
# > quit
```

---

## Decision Log

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2025-01-28 | Remove OpenAI entirely | Project rule: NO OpenAI | Complete rebuild |
| 2025-01-28 | Use Cartesia for voice | Unified TTS/STT, better latency | New `voice/` module |
| 2025-01-28 | Multi-LLM with router | Resilience, provider diversity | `llm/router.py` |
| 2025-01-28 | Three specialized agents | Domain expertise | Atlas, Cipher, Taskmaster |

---

## Open Questions

1. **Should we support OpenAI as optional LLM provider?**
   - Current: NO (violates project rule)
   - Consider: Make opt-in with explicit user request

2. **LiveKit vs. direct WebRTC?**
   - Current: LiveKit (abstraction layer)
   - Consider: Roll own WebRTC with aiortc for more control

3. **SQLite vs. PostgreSQL?**
   - Current: SQLite (simple, embedded)
   - Consider: PostgreSQL if multi-user deployment needed

4. **Streaming TTS: Buffer by sentence or fixed tokens?**
   - Current: Not implemented
   - Consider: Sentence-based (more natural) vs. fixed-size (lower latency)

---

## Next Action

**Immediate:** Plan Session 3 feature - likely **Streaming Voice Responses**

**Steps:**
1. Create PRP: `PRPs/2025-11-30_streaming_voice_responses.md`
2. Review architecture in `PLANNING.md`
3. Execute PRP using `.claude/commands/execute-prp.md`
4. Test with `python main.py --test`
5. Update `CLAUDE.md` with Session 3 summary

---

*Last Updated: 2025-11-30*
*Current Focus: Session 2 Complete - Planning Session 3*
