---
description: "Execute a Project Requirements Plan (PRP) with 6-phase workflow"
---

# Execute PRP for langgraph-voice-agents

**Critical Rules:**
- **NO OpenAI** - Use Claude, Gemini, OpenRouter only
- **Cartesia for all voice** - TTS (sonic-2) and STT (ink-whisper)
- API keys in `.env` only
- Test after EACH phase, not just at the end

---

## 6-Phase Execution Workflow

### Phase 0: Pre-Execution Checklist

Before starting, verify:

```bash
# 1. PRP file exists
ls /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/PRPs/*.md

# 2. Environment is clean
cd /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents
git status

# 3. Virtual environment active
which python  # Should show venv path

# 4. Current tests pass
python main.py --test
```

**Confirmation:**
- [ ] PRP reviewed and understood
- [ ] All dependencies listed in PRP
- [ ] Target files identified
- [ ] No merge conflicts in git
- [ ] Baseline tests passing

---

### Phase 1: Environment Setup

**Goal:** Prepare dependencies and configuration

#### 1.1 Add Dependencies (if any)

If PRP lists new packages:
```bash
# Add to requirements.txt
echo "new-package>=1.0.0  # Feature: [reason]" >> requirements.txt

# Install
pip install -r requirements.txt
```

#### 1.2 Update Environment Variables

If PRP requires new keys:
```bash
# Add to .env.example
echo "NEW_API_KEY=your_key_here  # Feature: [purpose]" >> .env.example

# Remind user to set in .env
echo "⚠️  Don't forget to set NEW_API_KEY in .env"
```

#### 1.3 Create File Structure

```bash
# Create new files listed in PRP
touch /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/path/to/new_file.py

# Create test files
touch /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/tests/test_new_feature.py
```

**Phase 1 Validation:**
```bash
# Verify imports work
python -c "import sys; sys.path.insert(0, '.'); import new_module"

# Check no syntax errors
python -m py_compile path/to/new_file.py
```

---

### Phase 2: Core Implementation

**Goal:** Implement main feature logic using established patterns

#### 2.1 VoiceAgent Pattern (if creating new agent)

```python
# File: agents/new_agent.py

from agents import VoiceAgent, VoiceAgentConfig
from core import AgentMode
from llm import LLMProvider, TaskType

class NewAgent(VoiceAgent):
    """
    [Description from PRP]

    NO OpenAI - Uses Cartesia for voice, Claude/Gemini/OpenRouter for LLM.
    """

    def __init__(self, voice_mode: bool = True):
        config = VoiceAgentConfig(
            agent_name="[Agent Name]",
            agent_description="[Description]",
            task_type=TaskType.[TASK_TYPE],  # From PRP
            default_provider=LLMProvider.[PROVIDER],  # From PRP
            greeting_message="[Optional greeting]",
        )

        super().__init__(
            config=config,
            mode=AgentMode.VOICE if voice_mode else AgentMode.TEXT
        )

    async def process_input(self, input_data: str) -> str:
        """Override to add custom processing"""
        # Custom logic here
        response = await super().process_input(input_data)
        return response
```

#### 2.2 LLM Integration Pattern

```python
# Use LLM router with automatic failover

async def intelligent_query(self, query: str) -> str:
    """
    Process query with optimal LLM provider.
    Automatically falls back if primary fails.
    """
    try:
        response = await self.generate_response(
            user_message=query,
            task_type=TaskType.[FROM_PRP],  # GENERAL, CODING, FAST, etc.
            provider=LLMProvider.[FROM_PRP],  # Optional: Claude, Gemini, OpenRouter
        )
        return response
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "I'm experiencing technical difficulties. Please try again."
```

#### 2.3 Voice Integration Pattern (if needed)

```python
# Cartesia TTS/STT integration

async def handle_voice_interaction(self, audio_data: bytes):
    """
    Process voice input and respond with voice.
    Uses Cartesia for ALL voice operations.
    """
    # STT: Cartesia ink-whisper
    transcription = await self.listen(audio_data)

    if not transcription.strip():
        return

    # Process with LLM
    response = await self.process_input(transcription)

    # TTS: Cartesia sonic-2
    if self.mode == AgentMode.VOICE:
        await self.speak(response)
```

**Phase 2 Validation:**
```bash
# Test imports
python -c "from agents import NewAgent; print('✅ Import works')"

# Test instantiation
python -c "
from agents import NewAgent
import asyncio

async def test():
    agent = NewAgent(voice_mode=False)
    print(f'✅ Created: {agent.config.agent_name}')
    await agent.close()

asyncio.run(test())
"
```

---

### Phase 3: Integration with Existing System

**Goal:** Wire up new feature to main CLI and other agents

#### 3.1 Update `agents/__init__.py`

```python
# Add to exports
from agents.new_agent import NewAgent

def create_new_agent(voice_mode: bool = True) -> NewAgent:
    """Factory function for NewAgent"""
    return NewAgent(voice_mode=voice_mode)

__all__ = [
    # ... existing exports ...
    "NewAgent",
    "create_new_agent",
]
```

#### 3.2 Update `main.py`

```python
# Add to imports
from agents import (
    create_general_assistant,
    create_code_assistant,
    create_task_manager,
    create_new_agent,  # NEW
)

# Add to select_agent() function
def select_agent() -> str:
    print("\n🤖 Available Agents:")
    print("-" * 40)
    # ... existing agents ...
    print("4. New Agent (NewName)")  # NEW
    print("   - [Description from PRP]")
    print()
    # ... rest of function ...

    while True:
        choice = input("\nSelect agent (1/2/3/4) or 'q' to quit: ").strip().lower()
        # ... existing choices ...
        elif choice == '4' or choice == 'new':  # NEW
            return 'new'
        # ... rest of function ...

# Add to main() function
def main():
    # ... existing code ...

    # In agent creation section:
    if agent_type == 'new':  # NEW
        agent = create_new_agent(voice_mode=voice_mode)
    # ... rest of function ...
```

#### 3.3 Update Integration Tests

Add test to `main.py` `run_tests()` function:

```python
async def run_tests():
    # ... existing tests ...

    # Test N: New Agent Creation
    print(f"\n{N}. Testing New Agent creation...")
    try:
        agent = create_new_agent(voice_mode=False)
        print(f"   ✅ Created: {agent.config.agent_name}")
        tests_passed += 1
        await agent.close()
    except Exception as e:
        print(f"   ❌ New Agent failed: {e}")
        tests_failed += 1
```

**Phase 3 Validation:**
```bash
# Run full integration test suite
python /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/main.py --test

# Test new agent selection
python /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents/main.py --agent new --text
```

---

### Phase 4: Voice & LLM Testing

**Goal:** Verify Cartesia voice and multi-LLM support

#### 4.1 Test LLM Provider Fallback

```python
# Create test script: test_llm_fallback.py
import asyncio
from agents import create_new_agent

async def test():
    agent = create_new_agent(voice_mode=False)

    # Test with each provider
    from llm import LLMProvider

    for provider in [LLMProvider.CLAUDE, LLMProvider.GEMINI, LLMProvider.OPENROUTER]:
        try:
            print(f"\nTesting {provider.value}...")
            response = await agent.generate_response(
                user_message="Say 'OK' if you're working",
                provider=provider
            )
            print(f"✅ {provider.value}: {response[:30]}...")
        except Exception as e:
            print(f"⚠️  {provider.value} unavailable: {e}")

    await agent.close()

asyncio.run(test())
```

#### 4.2 Test Cartesia Voice (if applicable)

```bash
# Test voice mode initialization
python -c "
import asyncio
import os
from agents import create_new_agent

async def test():
    if not os.getenv('CARTESIA_API_KEY'):
        print('⚠️  CARTESIA_API_KEY not set, skipping voice test')
        return

    agent = create_new_agent(voice_mode=True)
    print(f'Agent: {agent.config.agent_name}')
    print(f'Voice enabled: {agent.voice_enabled}')

    # Test TTS
    success = await agent.speak('Testing Cartesia sonic-2 TTS')
    print(f'TTS test: {'✅' if success else '❌'}')

    await agent.close()

asyncio.run(test())
"
```

**Phase 4 Validation:**
```bash
# Verify NO OpenAI dependencies
grep -ri "openai" /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents --exclude-dir=.git --exclude-dir=.claude

# Should return EMPTY or only in comments/docs
```

---

### Phase 5: Documentation

**Goal:** Update all project documentation

#### 5.1 Update `CLAUDE.md`

```markdown
# Add to "Agents" section:

| Agent | Name | Specialization | Default LLM |
|-------|------|----------------|-------------|
| ... existing agents ...
| `new_agent.py` | NewName | [Description] | [Provider] |

# Add to "Session History":

| Session | Date | Summary |
|---------|------|---------|
| ... existing sessions ...
| N | YYYY-MM-DD | Added NewAgent with [key features] |
```

#### 5.2 Update `README.md`

```markdown
# Add to usage examples:

## NewAgent Usage

```bash
python main.py --agent new
```

**Features:**
- [Feature 1 from PRP]
- [Feature 2 from PRP]

**LLM Provider:** [Claude/Gemini/OpenRouter]
**Voice:** Cartesia (sonic-2 TTS, ink-whisper STT)
```

#### 5.3 Update `.env.example` (if new vars)

```bash
# Add documentation for new variables
echo "" >> .env.example
echo "# NewAgent Configuration" >> .env.example
echo "NEW_VAR=value  # Purpose: [description]" >> .env.example
```

**Phase 5 Validation:**
```bash
# Verify markdown formatting
# (no specific command, visual check)

# Ensure .env.example has all required vars
diff <(grep "^[A-Z]" .env.example | cut -d= -f1 | sort) \
     <(grep "os.getenv" -r agents/ llm/ core/ voice/ | grep -o '"[A-Z_]*"' | tr -d '"' | sort -u)
```

---

### Phase 6: Final Validation

**Goal:** Comprehensive end-to-end testing

#### 6.1 Run Full Test Suite

```bash
cd /Users/tmkipper/Desktop/tk_projects/langgraph-voice-agents

# Integration tests
python main.py --test

# Code quality
black --check .
mypy . --ignore-missing-imports

# Environment check
python main.py  # Should show new agent in list
```

#### 6.2 Interactive Testing Checklist

Start the new agent and test:

```bash
python main.py --agent new
```

**Test Cases:**
- [ ] Agent starts without errors
- [ ] Responds to basic queries
- [ ] `voice on` enables voice mode (if Cartesia configured)
- [ ] `voice off` disables voice mode
- [ ] `clear` resets conversation history
- [ ] `quit` exits gracefully
- [ ] LLM responses are coherent
- [ ] Voice synthesis works (if in voice mode)

#### 6.3 Verify PRP Requirements Met

Go through PRP section by section:

- [ ] All "Technical Requirements" implemented
- [ ] All "Implementation Plan" phases completed
- [ ] All "File Changes" applied
- [ ] All "Validation Steps" pass
- [ ] NO OpenAI dependencies introduced
- [ ] Cartesia used for ALL voice operations
- [ ] At least one LLM provider works

#### 6.4 Git Commit (if all tests pass)

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add [Feature Name] to [Agent]

- [Key change 1]
- [Key change 2]
- Uses Cartesia for voice (sonic-2 TTS, ink-whisper STT)
- NO OpenAI dependencies

Implements: PRPs/YYYY-MM-DD_feature_name.md"

# Optionally push
git push origin main
```

---

## Execution Checklist

Complete workflow:

### Pre-Execution
- [ ] PRP file reviewed
- [ ] Baseline tests pass
- [ ] Git status clean

### Phase 1: Setup
- [ ] Dependencies added
- [ ] Environment variables configured
- [ ] File structure created
- [ ] Imports verified

### Phase 2: Core Implementation
- [ ] VoiceAgent pattern followed (if applicable)
- [ ] LLM integration implemented
- [ ] Voice integration implemented (if applicable)
- [ ] Code compiles without errors

### Phase 3: Integration
- [ ] `agents/__init__.py` updated
- [ ] `main.py` updated
- [ ] Integration tests added
- [ ] New agent selectable from CLI

### Phase 4: Voice & LLM Testing
- [ ] LLM provider fallback tested
- [ ] Cartesia voice tested (if applicable)
- [ ] NO OpenAI dependencies verified

### Phase 5: Documentation
- [ ] `CLAUDE.md` updated
- [ ] `README.md` updated
- [ ] `.env.example` updated (if needed)
- [ ] Code comments added

### Phase 6: Final Validation
- [ ] Full test suite passes
- [ ] Code quality checks pass
- [ ] Interactive testing complete
- [ ] PRP requirements verified
- [ ] Git commit created

---

## Rollback Procedure

If validation fails at any phase:

```bash
# 1. Stash changes
git stash

# 2. Verify baseline works
python main.py --test

# 3. Review errors
git stash show -p

# 4. Fix issues and retry
git stash pop

# 5. Repeat failed phase
```

---

## Critical Reminders

### ❌ NEVER
- Skip phase validation - test after EACH phase
- Use OpenAI for voice or LLM
- Hardcode API keys
- Commit without running `python main.py --test`

### ✅ ALWAYS
- Test incrementally (after each phase)
- Use Cartesia for voice (sonic-2, ink-whisper)
- Use LLMRouter with fallback
- Update documentation as you go
- Verify NO OpenAI dependencies at end

---

*Last Updated: 2025-11-30*
