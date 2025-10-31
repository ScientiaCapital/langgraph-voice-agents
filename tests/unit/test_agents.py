"""
Unit tests for agent classes
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from core import AgentMode
from agents import TaskOrchestrator, TaskExecutor, TaskChecker


@pytest.mark.unit
class TestTaskOrchestrator:
    """Unit tests for TaskOrchestrator"""

    def test_init_text_mode(self):
        """Test TaskOrchestrator initialization in TEXT mode"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)

        assert orchestrator.agent_type == "task-orchestrator"
        assert orchestrator.mode == AgentMode.TEXT
        assert orchestrator.livekit_client is None
        assert orchestrator.session_id is not None

    def test_init_with_checkpointer_path(self):
        """Test TaskOrchestrator with custom checkpointer path"""
        path = ":memory:test"
        orchestrator = TaskOrchestrator(
            mode=AgentMode.TEXT,
            checkpointer_path=path
        )

        assert orchestrator is not None
        assert orchestrator.checkpointer is not None

    def test_has_mcp_tools(self):
        """Test TaskOrchestrator has MCP tool adapters"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)

        assert hasattr(orchestrator, 'sequential_thinking')
        assert hasattr(orchestrator, 'serena')
        assert hasattr(orchestrator, 'context7')
        assert hasattr(orchestrator, 'taskmaster')
        assert hasattr(orchestrator, 'shrimp')

    def test_has_orchestration_state(self):
        """Test TaskOrchestrator has orchestration state management"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)

        assert hasattr(orchestrator, 'active_plans')
        assert hasattr(orchestrator, 'delegated_tasks')
        assert hasattr(orchestrator, 'agent_capabilities')
        assert isinstance(orchestrator.active_plans, dict)
        assert isinstance(orchestrator.delegated_tasks, dict)

    def test_has_voice_commands(self):
        """Test TaskOrchestrator has voice command handlers"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)

        assert hasattr(orchestrator, 'voice_commands')
        assert isinstance(orchestrator.voice_commands, dict)
        assert "plan task" in orchestrator.voice_commands
        assert "delegate task" in orchestrator.voice_commands

    def test_create_initial_state(self):
        """Test TaskOrchestrator can create initial state"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        task = "Test orchestration task"

        state = orchestrator.create_initial_state(task)

        assert state.agent_type == "task-orchestrator"
        assert state.current_task == task
        assert state.mode == AgentMode.TEXT

    @pytest.mark.asyncio
    async def test_process_input_string(self):
        """Test processing string input"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        result = await orchestrator.process_input("Test task")

        assert isinstance(result, dict)
        assert "status" in result
        assert "agent_type" in result

    @pytest.mark.asyncio
    async def test_process_input_dict(self):
        """Test processing dictionary input"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        result = await orchestrator.process_input({"task": "Test task"})

        assert isinstance(result, dict)
        assert "status" in result


@pytest.mark.unit
class TestTaskExecutor:
    """Unit tests for TaskExecutor"""

    def test_init_text_mode(self):
        """Test TaskExecutor initialization in TEXT mode"""
        executor = TaskExecutor(mode=AgentMode.TEXT)

        assert executor.agent_type == "task-executor"
        assert executor.mode == AgentMode.TEXT
        assert executor.livekit_client is None

    def test_has_mcp_tools(self):
        """Test TaskExecutor has MCP tool adapters"""
        executor = TaskExecutor(mode=AgentMode.TEXT)

        assert hasattr(executor, 'sequential_thinking')
        assert hasattr(executor, 'serena')
        assert hasattr(executor, 'context7')
        assert hasattr(executor, 'desktop_commander')
        assert hasattr(executor, 'shrimp')

    def test_has_execution_state(self):
        """Test TaskExecutor has execution state management"""
        executor = TaskExecutor(mode=AgentMode.TEXT)

        assert hasattr(executor, 'current_execution')
        assert hasattr(executor, 'active_processes')
        assert hasattr(executor, 'code_templates')
        assert isinstance(executor.active_processes, dict)
        assert isinstance(executor.code_templates, dict)

    def test_has_voice_commands(self):
        """Test TaskExecutor has voice command handlers"""
        executor = TaskExecutor(mode=AgentMode.TEXT)

        assert hasattr(executor, 'voice_commands')
        assert isinstance(executor.voice_commands, dict)
        assert "start implementation" in executor.voice_commands
        assert "run tests" in executor.voice_commands

    def test_create_initial_state(self):
        """Test TaskExecutor can create initial state"""
        executor = TaskExecutor(mode=AgentMode.TEXT)
        task = "Test execution task"

        state = executor.create_initial_state(task)

        assert state.agent_type == "task-executor"
        assert state.current_task == task

    @pytest.mark.asyncio
    async def test_process_input(self):
        """Test TaskExecutor processes input"""
        executor = TaskExecutor(mode=AgentMode.TEXT)
        result = await executor.process_input("Implement feature X")

        assert isinstance(result, dict)
        assert "agent_type" in result
        assert result["agent_type"] == "task-executor"


@pytest.mark.unit
class TestTaskChecker:
    """Unit tests for TaskChecker"""

    def test_init_text_mode(self):
        """Test TaskChecker initialization in TEXT mode"""
        checker = TaskChecker(mode=AgentMode.TEXT)

        assert checker.agent_type == "task-checker"
        assert checker.mode == AgentMode.TEXT
        assert checker.livekit_client is None

    def test_has_mcp_tools(self):
        """Test TaskChecker has MCP tool adapters"""
        checker = TaskChecker(mode=AgentMode.TEXT)

        assert hasattr(checker, 'sequential_thinking')
        assert hasattr(checker, 'serena')
        assert hasattr(checker, 'context7')
        assert hasattr(checker, 'taskmaster')
        assert hasattr(checker, 'shrimp')
        assert hasattr(checker, 'desktop_commander')

    def test_has_validation_state(self):
        """Test TaskChecker has validation state management"""
        checker = TaskChecker(mode=AgentMode.TEXT)

        assert hasattr(checker, 'validation_results')
        assert hasattr(checker, 'quality_metrics')
        assert hasattr(checker, 'current_validation_level')
        assert isinstance(checker.validation_results, list)

    def test_create_initial_state(self):
        """Test TaskChecker can create initial state"""
        checker = TaskChecker(mode=AgentMode.TEXT)
        task = "Validate implementation"

        state = checker.create_initial_state(task)

        assert state.agent_type == "task-checker"
        assert state.current_task == task

    @pytest.mark.asyncio
    async def test_process_input(self):
        """Test TaskChecker processes input"""
        checker = TaskChecker(mode=AgentMode.TEXT)
        result = await checker.process_input("Validate feature X")

        assert isinstance(result, dict)
        assert "agent_type" in result
        assert result["agent_type"] == "task-checker"


@pytest.mark.unit
class TestAgentFactoryFunctions:
    """Test agent factory functions"""

    def test_create_task_orchestrator(self):
        """Test create_task_orchestrator factory function"""
        from agents.task_orchestrator import create_task_orchestrator

        agent = create_task_orchestrator(mode=AgentMode.TEXT)

        assert agent is not None
        assert agent.agent_type == "task-orchestrator"

    def test_create_task_executor(self):
        """Test create_task_executor factory function"""
        from agents.task_executor import create_task_executor

        agent = create_task_executor(mode=AgentMode.TEXT)

        assert agent is not None
        assert agent.agent_type == "task-executor"
