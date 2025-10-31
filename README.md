# LangGraph Voice-Enabled Agent Framework

A comprehensive voice-enabled multi-agent framework built on LangGraph with LiveKit integration and extensive MCP (Model Context Protocol) tool adapters.

## 🎯 Overview

This framework provides a complete solution for building sophisticated voice-enabled AI agents that can handle complex development workflows through natural speech interaction. The system combines LangGraph's powerful graph-based agent orchestration with real-time voice communication via LiveKit.

## 🏗️ Architecture

### Core Components

- **Voice Integration**: LiveKit WebRTC for real-time audio processing
- **Agent Framework**: LangGraph StateGraph for workflow orchestration
- **MCP Tools**: Comprehensive tool adapters for development workflows
- **Multi-Modal Support**: Seamless text and voice interaction modes

### Agent Ecosystem

1. **Task Orchestrator Agent** (`agents/task_orchestrator.py`)
   - Strategic planning and task delegation
   - Complexity assessment and resource allocation
   - Cross-agent coordination and monitoring

2. **Task Executor Agent** (`agents/task_executor.py`)
   - Implementation and development execution
   - Comprehensive testing and validation
   - Code optimization and documentation

3. **Task Checker Agent** (`agents/task_checker.py`)
   - Quality assurance and validation
   - Multi-level testing strategies
   - Performance and security assessment

## 🛠️ MCP Tool Adapters

### Mandatory Tool Order
All agents follow the systematic MCP tool order:
1. **Sequential Thinking** → Problem decomposition and analysis
2. **Serena** → Code intelligence and project navigation
3. **Context7** → Best practices research and documentation

### Available Tools

- **Sequential Thinking** (`tools/sequential_thinking_tools.py`) - Systematic problem analysis
- **Serena** (`tools/serena_tools.py`) - Code intelligence and navigation
- **Context7** (`tools/context7_tools.py`) - Documentation and best practices
- **Taskmaster AI** (`tools/taskmaster_tools.py`) - Task management and research
- **Shrimp Task Manager** (`tools/shrimp_tools.py`) - Advanced task planning
- **Desktop Commander** (`tools/desktop_commander_tools.py`) - File system operations

## 🎙️ Voice Features

### Voice Commands
- **Strategic Commands**: "orchestrate project", "analyze complexity", "delegate tasks"
- **Implementation Commands**: "implement feature", "run tests", "optimize code"
- **Quality Commands**: "run validation", "security scan", "quality report"

### Voice Processing
- **Speech-to-Text**: OpenAI Whisper integration
- **Text-to-Speech**: OpenAI TTS with customizable voices
- **Real-time Audio**: LiveKit WebRTC communication
- **Voice Activity Detection**: Automatic speech detection and processing

## 🚀 Quick Start

### Installation

#### Option 1: Install from source (recommended for development)
```bash
# Clone the repository
git clone https://github.com/ScientiaCapital/langgraph-voice-agents.git
cd langgraph-voice-agents

# Install with development dependencies
make install-dev

# Or manually:
pip install -e ".[dev]"
```

#### Option 2: Install as package
```bash
pip install langgraph-voice-agents
```

### Environment Setup
```bash
# Create .env file from template
make env-setup

# Or manually:
cp .env.example .env

# Edit .env with your API keys
# At minimum: OPENAI_API_KEY, LANGCHAIN_API_KEY
```

### Run Your First Demo
```bash
# Run basic demo (no LiveKit required)
make run-demo

# Or manually:
python examples/basic_demo.py
```

### Verify Installation
```bash
# Run structural validation
make structure-test

# Run all tests (if dependencies installed)
make test
```

### Basic Usage
```python
import asyncio
from agents import TaskOrchestrator, TaskExecutor, TaskChecker
from core import AgentMode

async def main():
    # Create agents in TEXT mode (no voice required)
    orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
    executor = TaskExecutor(mode=AgentMode.TEXT)
    checker = TaskChecker(mode=AgentMode.TEXT)

    # Process a task
    task = "Build a REST API with JWT authentication"

    # Step 1: Plan with orchestrator
    plan = await orchestrator.process_input(task)
    print(f"Plan: {plan['status']}")

    # Step 2: Execute with executor
    result = await executor.process_input(task)
    print(f"Execution: {result['status']}")

    # Step 3: Validate with checker
    validation = await checker.process_input(f"Validate: {task}")
    print(f"Validation: {validation['status']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Voice-Enabled Usage (requires LiveKit)
```python
from agents import TaskOrchestrator
from voice import LiveKitConfig
from core import AgentMode
import os

# Configure LiveKit
config = LiveKitConfig(
    url=os.getenv('LIVEKIT_URL'),
    api_key=os.getenv('LIVEKIT_API_KEY'),
    api_secret=os.getenv('LIVEKIT_API_SECRET'),
    room_name="agent-session"
)

# Create voice-enabled agent
agent = TaskOrchestrator(
    mode=AgentMode.VOICE,
    livekit_config=config
)

# Process voice or text input
result = await agent.process_input("Design a microservices architecture")
```

## 📋 Common Commands

```bash
# Development
make install          # Install package
make install-dev      # Install with dev dependencies
make clean            # Clean build artifacts

# Testing
make test             # Run all tests
make test-unit        # Unit tests only
make test-integration # Integration tests only
make lint             # Run linting
make format           # Format code

# Usage
make run-demo         # Run basic demo
make run-voice-demo   # Run voice demo (requires LiveKit)
make validate         # Run all validation checks

# Setup
make dev-setup        # Complete dev environment setup
make env-setup        # Create .env from template
make env-check        # Check environment variables

# Information
make help             # Show all commands
make info             # Project information
```

## 📁 Project Structure

```
langgraph-voice-agents/
├── agents/                     # Core agent implementations
│   ├── __init__.py             # Agent exports
│   ├── task_orchestrator.py   # Strategic planning agent
│   ├── task_executor.py       # Implementation agent
│   └── task_checker.py        # Quality assurance agent
├── core/                       # Framework foundation
│   ├── __init__.py             # Core exports
│   ├── base_graph.py          # BaseAgent, AgentState, AgentMode
│   └── state_management.py    # StateManager, ExecutionState
├── tools/                      # MCP tool adapters
│   ├── __init__.py             # Tool exports
│   ├── sequential_thinking_tools.py
│   ├── serena_tools.py
│   ├── context7_tools.py
│   ├── taskmaster_tools.py
│   ├── shrimp_tools.py
│   └── desktop_commander_tools.py
├── voice/                      # Voice integration
│   ├── __init__.py             # Voice exports
│   └── livekit_client.py      # LiveKit WebRTC client
├── examples/                   # Usage examples
│   ├── basic_demo.py           # Basic agent usage
│   ├── voice_demo.py           # Voice-enabled demo
│   ├── error_handling/         # Error handling examples
│   └── README.md               # Examples documentation
├── tests/                      # Test suites
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── voice/                  # Voice tests
│   ├── conftest.py             # Pytest configuration
│   └── README.md               # Testing documentation
├── docs/                       # Documentation
│   └── API_REFERENCE.md        # Complete API reference
├── .env.example                # Environment template
├── Makefile                    # Development commands
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Package configuration
├── requirements.txt            # Dependencies
├── test_structure.py           # Structural validation
├── test_imports.py             # Import smoke tests
├── README.md                   # This file
└── CLAUDE.md                   # Development log
```

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI API (for voice processing)
OPENAI_API_KEY=your_openai_api_key

# LiveKit (for voice communication)
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# MCP Tool Configuration
TASKMASTER_API_KEY=your_taskmaster_key
SERENA_PROJECT_PATH=/path/to/your/project
CONTEXT7_API_KEY=your_context7_key
```

### Agent Configuration
```python
# Customize agent behavior
agent_config = {
    "validation_level": "comprehensive",
    "voice_enabled": True,
    "auto_delegation": True,
    "quality_threshold": 85.0
}
```

## 🧪 Testing

### Run Tests
```bash
# All tests with coverage
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Voice tests (requires LiveKit setup)
make test-voice

# Fast tests (no coverage)
make test-fast

# Specific test file
pytest tests/unit/test_core.py -v

# Specific test function
pytest tests/integration/test_agents.py::TestAgentInstantiation::test_task_orchestrator_creation -v
```

### Test Markers
```bash
# Run only unit tests
pytest -m "unit"

# Run everything except voice tests
pytest -m "not voice"

# Run fast tests only (exclude slow)
pytest -m "not slow"
```

### Coverage Reports
```bash
# Terminal coverage report
pytest --cov=. --cov-report=term-missing

# HTML coverage report
pytest --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

See [tests/README.md](tests/README.md) for more testing information.

## 📚 Documentation

### Core Documentation
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation for all components
- **[Examples](examples/README.md)** - Usage examples and tutorials
- **[Testing Guide](tests/README.md)** - Comprehensive testing documentation
- **[Error Handling](examples/error_handling/README.md)** - Error handling patterns and best practices

### Quick Links
- **Basic Demo**: [examples/basic_demo.py](examples/basic_demo.py)
- **Voice Demo**: [examples/voice_demo.py](examples/voice_demo.py)
- **Error Examples**: [examples/error_handling/error_handling_demo.py](examples/error_handling/error_handling_demo.py)
- **Development Log**: [CLAUDE.md](CLAUDE.md)

### Component Documentation
- **Agents**: See [docs/API_REFERENCE.md#agents](docs/API_REFERENCE.md#agents)
  - TaskOrchestrator - Strategic planning
  - TaskExecutor - Implementation
  - TaskChecker - Quality assurance
- **Core**: See [docs/API_REFERENCE.md#core-module](docs/API_REFERENCE.md#core-module)
  - BaseAgent - Abstract base class
  - AgentState - State dataclass
  - AgentMode - Execution modes
  - StateManager - State persistence
- **Tools**: See [docs/API_REFERENCE.md#tools](docs/API_REFERENCE.md#tools)
  - 6 MCP tool adapters for various capabilities
- **Voice**: See [docs/API_REFERENCE.md#voice](docs/API_REFERENCE.md#voice)
  - LiveKitClient - Voice integration

## 🎯 Key Features

### Multi-Modal Interaction
- **Voice-First Design**: Natural speech interaction for all workflows
- **Text Fallback**: Full text-based operation when voice unavailable
- **Context Switching**: Seamless mode transitions during operation

### Advanced Orchestration
- **Dependency Management**: Automatic task dependency resolution
- **Resource Allocation**: Intelligent agent and tool selection
- **Progress Monitoring**: Real-time workflow tracking and reporting

### Quality Assurance
- **Multi-Level Validation**: Basic to enterprise-grade quality checks
- **Automated Testing**: Comprehensive test execution and reporting
- **Performance Monitoring**: Real-time metrics and optimization

### State Management
- **Persistent State**: SQLite/Redis hybrid storage
- **Session Recovery**: Automatic state restoration
- **Cross-Agent Sync**: Shared state across agent instances

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **LangGraph**: Graph-based agent framework foundation
- **LiveKit**: Real-time voice communication infrastructure
- **MCP Protocol**: Model Context Protocol for tool integration
- **OpenAI**: Speech processing and AI capabilities

## 🔗 Links

- [GitHub Repository](https://github.com/ScientiaCapital/langgraph-voice-agents)
- [Documentation](https://github.com/ScientiaCapital/langgraph-voice-agents/docs)
- [Issues](https://github.com/ScientiaCapital/langgraph-voice-agents/issues)
- [Discussions](https://github.com/ScientiaCapital/langgraph-voice-agents/discussions)

---

Built with ❤️ for the future of voice-enabled AI development.