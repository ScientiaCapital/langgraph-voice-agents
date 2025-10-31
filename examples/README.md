# Examples - LangGraph Voice Agent Framework

This directory contains practical examples demonstrating how to use the LangGraph Voice-Enabled Agent Framework.

## Available Examples

### 1. Basic Demo (`basic_demo.py`)

**Purpose:** Demonstrates core agent functionality without voice capabilities.

**What it shows:**
- Creating and initializing agents
- Processing tasks with TaskOrchestrator
- Implementing features with TaskExecutor
- Validating work with TaskChecker
- Agent collaboration workflow

**Requirements:**
- Python 3.9+
- Core dependencies installed (`pip install -e .`)
- No MCP servers required (agents will work without them)
- No LiveKit required (uses TEXT mode)

**Usage:**
```bash
python examples/basic_demo.py
```

**Expected output:**
```
Demo 1: Task Orchestrator - Strategic Planning
✓ TaskOrchestrator demo completed successfully

Demo 2: Task Executor - Implementation & Execution
✓ TaskExecutor demo completed successfully

Demo 3: Task Checker - Quality Assurance & Validation
✓ TaskChecker demo completed successfully

Demo 4: Agent Collaboration - Complete Workflow
✓ Agent collaboration demo completed successfully

Total: 4/4 demos completed successfully
🎉 All demos passed! Framework is working correctly.
```

---

### 2. Voice Demo (`voice_demo.py`)

**Purpose:** Demonstrates voice-enabled agent interactions using LiveKit.

**What it shows:**
- LiveKit configuration and setup
- Voice-enabled agent creation (AgentMode.VOICE)
- Real-time audio processing pipeline
- Speech-to-text (Whisper) and text-to-speech (TTS) integration

**Requirements:**
- All basic demo requirements
- LiveKit server running
- Environment variables configured:
  - `LIVEKIT_URL`
  - `LIVEKIT_API_KEY`
  - `LIVEKIT_API_SECRET`
  - `OPENAI_API_KEY`

**Usage:**
```bash
# Ensure .env is configured
cp .env.example .env
# Edit .env with your credentials

python examples/voice_demo.py
```

---

## Quick Start Guide

### Step 1: Install Dependencies

```bash
# From project root
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
# At minimum, set:
# - OPENAI_API_KEY (for MCP tools that may use LLMs)
```

### Step 3: Run Basic Demo

```bash
python examples/basic_demo.py
```

If this passes, your installation is correct!

---

## Understanding Agent Modes

The framework supports three agent modes:

### TEXT Mode (Default)
```python
agent = TaskOrchestrator(mode=AgentMode.TEXT)
```
- No voice capabilities
- Pure text input/output
- Fastest and simplest
- No LiveKit required

### VOICE Mode
```python
agent = TaskOrchestrator(
    mode=AgentMode.VOICE,
    livekit_config=livekit_config
)
```
- Full voice interaction
- Speech-to-text via Whisper
- Text-to-speech responses
- Requires LiveKit server

### HYBRID Mode
```python
agent = TaskOrchestrator(mode=AgentMode.HYBRID)
```
- Supports both text and voice
- Dynamically switches based on input
- Best of both worlds

---

## Example Workflows

### Simple Task Planning
```python
from agents import TaskOrchestrator
from core import AgentMode

# Create agent
orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)

# Plan a task
result = await orchestrator.process_input(
    "Design a caching layer for a high-traffic API"
)

print(result)
```

### Implementation Workflow
```python
from agents import TaskExecutor
from core import AgentMode

# Create agent
executor = TaskExecutor(mode=AgentMode.TEXT)

# Implement a feature
result = await executor.process_input(
    "Implement Redis caching with connection pooling"
)

print(result)
```

### Quality Validation
```python
from agents import TaskChecker
from core import AgentMode

# Create agent
checker = TaskChecker(mode=AgentMode.TEXT)

# Validate implementation
result = await checker.process_input(
    "Validate Redis caching implementation for thread safety"
)

print(result)
```

---

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'langgraph'
```
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Missing Environment Variables
```
KeyError: 'OPENAI_API_KEY'
```
**Solution:** Configure `.env` file based on `.env.example`

### LiveKit Connection Errors
```
ConnectionError: Failed to connect to LiveKit
```
**Solution:**
1. Ensure LiveKit server is running
2. Check `LIVEKIT_URL` is correct
3. Verify API credentials

### Abstract Method Errors
```
TypeError: Can't instantiate abstract class
```
**Solution:** This shouldn't happen after Option A fixes. Verify all structural fixes are in place.

---

## Next Steps

After running the examples:

1. **Explore MCP Tools:** Set up MCP tool servers for enhanced capabilities
2. **Build Custom Agents:** Extend BaseAgent to create specialized agents
3. **Add Voice:** Configure LiveKit for voice interactions
4. **Integrate Workflows:** Build multi-agent workflows for complex tasks

---

## Contributing

Found a bug in the examples? Have a suggestion for a new example?

1. Check existing issues
2. Open a new issue with details
3. Submit a PR with your improvements

---

## Support

For help and support:
- Check the main [README.md](../README.md)
- Review [CLAUDE.md](../CLAUDE.md) for development notes
- Open an issue on GitHub
