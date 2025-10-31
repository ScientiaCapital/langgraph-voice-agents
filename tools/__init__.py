"""
MCP (Model Context Protocol) tool adapters for agent capabilities.

This package provides adapters for various MCP servers including:
- Sequential Thinking: Problem decomposition and analysis
- Serena: Code intelligence and project navigation
- Context7: Documentation and best practices research
- TaskMaster: Task management with research capabilities
- Shrimp: Advanced task planning and verification
- Desktop Commander: File system operations and process management
"""

from .sequential_thinking_tools import SequentialThinkingAdapter
from .serena_tools import SerenaAdapter
from .context7_tools import Context7Adapter
from .taskmaster_tools import TaskMasterAdapter
from .shrimp_tools import ShrimpTaskManagerAdapter
from .desktop_commander_tools import DesktopCommanderAdapter

__all__ = [
    "SequentialThinkingAdapter",
    "SerenaAdapter",
    "Context7Adapter",
    "TaskMasterAdapter",
    "ShrimpTaskManagerAdapter",
    "DesktopCommanderAdapter",
]
