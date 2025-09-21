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

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage
```python
from agents.task_orchestrator import TaskOrchestratorAgent
from voice.livekit_client import create_livekit_client

# Create voice-enabled orchestrator
livekit_client = await create_livekit_client(
    agent_type="orchestrator",
    session_id="demo-session",
    participant_name="AI-Agent",
    livekit_url="wss://your-livekit-server.com",
    api_key="your-api-key",
    api_secret="your-api-secret",
    openai_api_key="your-openai-key"
)

agent = TaskOrchestratorAgent(
    session_id="demo-session",
    livekit_client=livekit_client
)

# Run via voice or text
result = await agent.run({
    "messages": [],
    "task_description": "Build a REST API with authentication"
})
```

## 📁 Project Structure

```
langgraph-flows/
├── agents/                 # Core agent implementations
│   ├── task_orchestrator.py   # Strategic planning agent
│   ├── task_executor.py       # Implementation agent
│   └── task_checker.py        # Quality assurance agent
├── core/                   # Framework foundation
│   ├── base_agent.py          # Base agent class
│   ├── base_graph.py          # Graph utilities
│   ├── multimodal_mixin.py    # Voice/text support
│   └── state_management.py    # State persistence
├── tools/                  # MCP tool adapters
│   ├── sequential_thinking_tools.py
│   ├── serena_tools.py
│   ├── context7_tools.py
│   ├── taskmaster_tools.py
│   ├── shrimp_tools.py
│   └── desktop_commander_tools.py
├── voice/                  # Voice integration
│   └── livekit_client.py      # LiveKit WebRTC client
├── examples/               # Usage examples
├── tests/                  # Test suites
└── patterns/               # Common workflows
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

Run the comprehensive test suite:
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Voice interaction tests
pytest tests/voice/

# Full test suite
pytest tests/ -v --cov=.
```

## 📚 Documentation

- **Agent Development Guide**: `docs/agent_development.md`
- **Voice Integration Guide**: `docs/voice_integration.md`
- **MCP Tool Reference**: `docs/mcp_tools.md`
- **API Documentation**: `docs/api_reference.md`

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