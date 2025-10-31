# API Reference - LangGraph Voice Agent Framework

Complete API reference for all framework components.

## Table of Contents

- [Core Module](#core-module)
  - [BaseAgent](#baseagent)
  - [AgentState](#agentstate)
  - [AgentMode](#agentmode)
  - [StateManager](#statemanager)
- [Agents](#agents)
  - [TaskOrchestrator](#taskorchestrator)
  - [TaskExecutor](#taskexecutor)
  - [TaskChecker](#taskchecker)
- [Tools](#tools)
  - [MCP Tool Adapters](#mcp-tool-adapters)
- [Voice](#voice)
  - [LiveKitClient](#livekitclient)

---

## Core Module

### BaseAgent

Abstract base class for all agents in the framework.

```python
from core import BaseAgent, AgentMode

class BaseAgent(ABC):
    def __init__(
        self,
        agent_type: str,
        mode: AgentMode = AgentMode.TEXT,
        checkpointer_path: Optional[str] = None
    )
```

**Parameters:**
- `agent_type` (str): Identifier for the agent type
- `mode` (AgentMode): Execution mode (TEXT, VOICE, or HYBRID)
- `checkpointer_path` (str, optional): Path for SQLite state persistence

**Abstract Methods:**
- `_build_graph() -> StateGraph`: Build the agent's LangGraph workflow
- `async process_input(input_data: Union[str, Dict]) -> Dict`: Process input and return result

**Methods:**
- `create_initial_state(task: str, mode: Optional[AgentMode]) -> AgentState`: Create initial state for a task

**Attributes:**
- `agent_type` (str): The agent's type identifier
- `mode` (AgentMode): Current execution mode
- `session_id` (str): Unique session identifier (UUID)
- `graph` (StateGraph): The compiled LangGraph
- `app`: The compiled application with checkpointer

**Example:**
```python
from agents import TaskOrchestrator
from core import AgentMode

# Create agent
agent = TaskOrchestrator(mode=AgentMode.TEXT)

# Process task
result = await agent.process_input("Design a REST API")
```

---

### AgentState

Data class representing the shared state for all agents.

```python
from core import AgentState, AgentMode

@dataclass
class AgentState:
    # Core identification
    session_id: str
    agent_type: str
    mode: AgentMode

    # Task management
    current_task: Optional[str] = None
    task_history: List[str] = None
    task_context: Dict[str, Any] = None

    # Communication
    messages: List[Dict[str, Any]] = None
    voice_input: Optional[str] = None
    voice_output: Optional[str] = None

    # MCP tool integration
    sequential_thinking_result: Optional[Dict] = None
    serena_analysis: Optional[Dict] = None
    context7_docs: Optional[Dict] = None

    # LiveKit session
    livekit_room_id: Optional[str] = None
    livekit_participant_id: Optional[str] = None

    # Error handling
    errors: List[str] = None
    retry_count: int = 0
```

**Fields:**
- **Core**: session_id, agent_type, mode
- **Tasks**: current_task, task_history, task_context
- **Communication**: messages, voice_input, voice_output
- **MCP Tools**: sequential_thinking_result, serena_analysis, context7_docs
- **LiveKit**: livekit_room_id, livekit_participant_id
- **Errors**: errors list, retry_count

**Example:**
```python
state = AgentState(
    session_id="session-123",
    agent_type="task-orchestrator",
    mode=AgentMode.TEXT,
    current_task="Build authentication API"
)
```

---

### AgentMode

Enum defining agent execution modes.

```python
from core import AgentMode

class AgentMode(Enum):
    TEXT = "text"      # Text-only mode
    VOICE = "voice"    # Voice-enabled mode
    HYBRID = "hybrid"  # Both text and voice
```

**Usage:**
```python
# Create text-only agent
agent = TaskOrchestrator(mode=AgentMode.TEXT)

# Create voice-enabled agent
agent = TaskOrchestrator(
    mode=AgentMode.VOICE,
    livekit_config=config
)
```

---

### StateManager

Manages persistent state using SQLite and optional Redis.

```python
from core import StateManager

class StateManager:
    def __init__(
        self,
        db_url: str = "sqlite:///agent_state.db",
        redis_url: Optional[str] = None
    )
```

**Parameters:**
- `db_url` (str): SQLAlchemy database URL
- `redis_url` (str, optional): Redis connection URL

**Example:**
```python
# SQLite only
manager = StateManager(db_url="sqlite:///state.db")

# With Redis
manager = StateManager(
    db_url="sqlite:///state.db",
    redis_url="redis://localhost:6379/0"
)
```

---

## Agents

### TaskOrchestrator

Strategic planning and task delegation agent.

```python
from agents import TaskOrchestrator
from core import AgentMode

class TaskOrchestratorAgent(BaseAgent):
    def __init__(
        self,
        mode: AgentMode = AgentMode.TEXT,
        checkpointer_path: Optional[str] = None,
        state_manager: Optional[StateManager] = None,
        livekit_config: Optional[LiveKitConfig] = None
    )
```

**Key Features:**
- Strategic planning using Sequential Thinking
- Task complexity analysis
- Sub-task delegation
- Multi-agent coordination

**MCP Tools:**
- Sequential Thinking (required)
- Serena (code intelligence)
- Context7 (documentation research)
- TaskMaster (task management)
- Shrimp (advanced planning)

**Voice Commands:**
- "plan task" - Create strategic plan
- "delegate task" - Delegate to sub-agents
- "check status" - Get progress status
- "emergency stop" - Halt all activities

**Example:**
```python
orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)

result = await orchestrator.process_input(
    "Design a microservices architecture for e-commerce"
)

print(result["status"])  # 'completed'
```

---

### TaskExecutor

Implementation and development execution agent.

```python
from agents import TaskExecutor
from core import AgentMode

class TaskExecutorAgent(BaseAgent):
    def __init__(
        self,
        mode: AgentMode = AgentMode.TEXT,
        checkpointer_path: Optional[str] = None,
        state_manager: Optional[StateManager] = None,
        livekit_config: Optional[LiveKitConfig] = None
    )
```

**Key Features:**
- Implementation execution
- Code generation and testing
- File system operations
- Process management

**MCP Tools:**
- Sequential Thinking (analysis)
- Serena (code intelligence)
- Context7 (best practices)
- Desktop Commander (file operations)
- Shrimp (task verification)

**Voice Commands:**
- "start implementation" - Begin implementation
- "run tests" - Execute test suite
- "check status" - Get implementation status
- "commit changes" - Commit code changes
- "debug issue" - Start debugging workflow
- "optimize code" - Code optimization workflow

**Example:**
```python
executor = TaskExecutor(mode=AgentMode.TEXT)

result = await executor.process_input(
    "Implement JWT authentication middleware"
)
```

---

### TaskChecker

Quality assurance and validation agent.

```python
from agents import TaskChecker
from core import AgentMode

class TaskCheckerAgent(BaseAgent):
    def __init__(
        self,
        mode: AgentMode = AgentMode.TEXT,
        checkpointer_path: Optional[str] = None,
        livekit_config: Optional[LiveKitConfig] = None
    )
```

**Key Features:**
- Comprehensive testing
- Code quality validation
- Security scanning
- Performance testing

**Validation Levels:**
- BASIC: Quick sanity checks
- STANDARD: Comprehensive testing
- COMPREHENSIVE: Full test suite
- ENTERPRISE: Production-grade validation

**MCP Tools:**
- Sequential Thinking (analysis)
- Serena (code review)
- Context7 (best practices)
- TaskMaster (test planning)
- Shrimp (verification)
- Desktop Commander (test execution)

**Example:**
```python
checker = TaskChecker(mode=AgentMode.TEXT)

result = await checker.process_input(
    "Validate authentication middleware implementation"
)
```

---

## Tools

### MCP Tool Adapters

#### SequentialThinkingAdapter
```python
from tools import SequentialThinkingAdapter

adapter = SequentialThinkingAdapter()
```
Systematic problem decomposition and analysis.

#### SerenaAdapter
```python
from tools import SerenaAdapter

adapter = SerenaAdapter()
```
Code intelligence and project navigation.

#### Context7Adapter
```python
from tools import Context7Adapter

adapter = Context7Adapter()
```
Documentation and best practices research.

#### TaskMasterAdapter
```python
from tools import TaskMasterAdapter

adapter = TaskMasterAdapter()
```
Task management with research capabilities.

#### ShrimpTaskManagerAdapter
```python
from tools import ShrimpTaskManagerAdapter

adapter = ShrimpTaskManagerAdapter()
```
Advanced task planning and verification.

#### DesktopCommanderAdapter
```python
from tools import DesktopCommanderAdapter

adapter = DesktopCommanderAdapter()
```
File system operations and process management.

---

## Voice

### LiveKitClient

LiveKit integration for real-time voice communication.

```python
from voice import LiveKitClient, LiveKitConfig

config = LiveKitConfig(
    url="wss://your-livekit-server.cloud",
    api_key="your_api_key",
    api_secret="your_api_secret",
    room_name="agent-session"
)

client = LiveKitClient(config)
```

**Features:**
- WebRTC real-time audio
- OpenAI Whisper for STT
- OpenAI TTS for voice synthesis
- Voice Activity Detection (VAD)

**Example:**
```python
from agents import TaskOrchestrator
from voice import LiveKitConfig
from core import AgentMode

config = LiveKitConfig(
    url=os.getenv('LIVEKIT_URL'),
    api_key=os.getenv('LIVEKIT_API_KEY'),
    api_secret=os.getenv('LIVEKIT_API_SECRET')
)

agent = TaskOrchestrator(
    mode=AgentMode.VOICE,
    livekit_config=config
)

# Now agent can process voice input
```

---

## Factory Functions

### create_task_orchestrator
```python
from agents.task_orchestrator import create_task_orchestrator

agent = create_task_orchestrator(
    mode=AgentMode.TEXT,
    checkpointer_path=":memory:",
    state_manager=state_manager,
    livekit_config=livekit_config
)
```

### create_task_executor
```python
from agents.task_executor import create_task_executor

agent = create_task_executor(
    mode=AgentMode.TEXT,
    state_manager=state_manager
)
```

---

## Error Handling

All agents handle errors gracefully and store them in the agent state:

```python
try:
    result = await agent.process_input(task)
except Exception as e:
    print(f"Error: {e}")
    # Errors are stored in agent state
    # Check agent.app state for error details
```

---

## Type Annotations

The framework is fully type-annotated for better IDE support:

```python
from typing import Dict, Any, Optional, Union

async def process_input(
    self,
    input_data: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    ...
```

---

## See Also

- [README.md](../README.md) - Project overview
- [Examples](../examples/) - Usage examples
- [Tests](../tests/) - Test suite documentation
- [CLAUDE.md](../CLAUDE.md) - Development log

---

*API Reference v0.1.0 - Last updated: 2025-01-21*
