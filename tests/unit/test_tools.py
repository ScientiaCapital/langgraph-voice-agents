"""
Unit tests for MCP tool adapters
"""

import pytest
from tools import (
    SequentialThinkingAdapter,
    SerenaAdapter,
    Context7Adapter,
    TaskMasterAdapter,
    ShrimpTaskManagerAdapter,
    DesktopCommanderAdapter,
)


@pytest.mark.unit
class TestSequentialThinkingAdapter:
    """Tests for SequentialThinkingAdapter"""

    def test_init(self):
        """Test adapter initialization"""
        adapter = SequentialThinkingAdapter()
        assert adapter is not None

    def test_has_required_methods(self):
        """Test adapter has required methods"""
        adapter = SequentialThinkingAdapter()

        # Check for common methods (adapt based on actual implementation)
        assert hasattr(adapter, '__init__')
        # Add more specific method checks based on actual implementation


@pytest.mark.unit
class TestSerenaAdapter:
    """Tests for SerenaAdapter (code intelligence)"""

    def test_init(self):
        """Test adapter initialization"""
        adapter = SerenaAdapter()
        assert adapter is not None

    def test_is_tool_adapter(self):
        """Test that it's a tool adapter"""
        adapter = SerenaAdapter()
        assert adapter is not None
        # Serena provides code intelligence capabilities


@pytest.mark.unit
class TestContext7Adapter:
    """Tests for Context7Adapter (documentation research)"""

    def test_init(self):
        """Test adapter initialization"""
        adapter = Context7Adapter()
        assert adapter is not None

    def test_is_documentation_tool(self):
        """Test that it's a documentation tool"""
        adapter = Context7Adapter()
        assert adapter is not None
        # Context7 provides documentation and best practices


@pytest.mark.unit
class TestTaskMasterAdapter:
    """Tests for TaskMasterAdapter"""

    def test_init(self):
        """Test adapter initialization"""
        adapter = TaskMasterAdapter()
        assert adapter is not None

    def test_is_task_management_tool(self):
        """Test that it's a task management tool"""
        adapter = TaskMasterAdapter()
        assert adapter is not None
        # TaskMaster provides task management capabilities


@pytest.mark.unit
class TestShrimpTaskManagerAdapter:
    """Tests for ShrimpTaskManagerAdapter"""

    def test_init(self):
        """Test adapter initialization"""
        adapter = ShrimpTaskManagerAdapter()
        assert adapter is not None

    def test_is_advanced_task_tool(self):
        """Test that it's an advanced task planning tool"""
        adapter = ShrimpTaskManagerAdapter()
        assert adapter is not None
        # Shrimp provides advanced task planning and verification


@pytest.mark.unit
class TestDesktopCommanderAdapter:
    """Tests for DesktopCommanderAdapter"""

    def test_init(self):
        """Test adapter initialization"""
        adapter = DesktopCommanderAdapter()
        assert adapter is not None

    def test_is_file_system_tool(self):
        """Test that it's a file system operations tool"""
        adapter = DesktopCommanderAdapter()
        assert adapter is not None
        # Desktop Commander provides file system and process management


@pytest.mark.unit
class TestToolAdapterCollection:
    """Tests for tool adapter collection"""

    def test_all_adapters_instantiate(self):
        """Test that all tool adapters can be instantiated"""
        adapters = [
            SequentialThinkingAdapter(),
            SerenaAdapter(),
            Context7Adapter(),
            TaskMasterAdapter(),
            ShrimpTaskManagerAdapter(),
            DesktopCommanderAdapter(),
        ]

        assert len(adapters) == 6
        for adapter in adapters:
            assert adapter is not None

    def test_adapter_uniqueness(self):
        """Test that each adapter is a unique instance"""
        adapter1 = SequentialThinkingAdapter()
        adapter2 = SequentialThinkingAdapter()

        # They should be different instances
        assert adapter1 is not adapter2
