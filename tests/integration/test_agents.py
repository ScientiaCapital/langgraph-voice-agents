"""
Integration tests for agent instantiation and basic workflows
"""

import pytest
from core import AgentMode
from agents import TaskOrchestrator, TaskExecutor, TaskChecker


@pytest.mark.integration
class TestAgentInstantiation:
    """Tests for agent creation and initialization"""

    def test_task_orchestrator_creation(self, agent_mode_text):
        """Test TaskOrchestrator can be instantiated"""
        agent = TaskOrchestrator(mode=agent_mode_text)

        assert agent is not None
        assert agent.agent_type == "task-orchestrator"
        assert agent.mode == AgentMode.TEXT
        assert agent.session_id is not None

    def test_task_executor_creation(self, agent_mode_text):
        """Test TaskExecutor can be instantiated"""
        agent = TaskExecutor(mode=agent_mode_text)

        assert agent is not None
        assert agent.agent_type == "task-executor"
        assert agent.mode == AgentMode.TEXT
        assert agent.session_id is not None

    def test_task_checker_creation(self, agent_mode_text):
        """Test TaskChecker can be instantiated"""
        agent = TaskChecker(mode=agent_mode_text)

        assert agent is not None
        assert agent.agent_type == "task-checker"
        assert agent.mode == AgentMode.TEXT
        assert agent.session_id is not None

    def test_multiple_agents_have_unique_sessions(self, agent_mode_text):
        """Test that multiple agents have unique session IDs"""
        agent1 = TaskOrchestrator(mode=agent_mode_text)
        agent2 = TaskOrchestrator(mode=agent_mode_text)

        assert agent1.session_id != agent2.session_id


@pytest.mark.integration
class TestAgentStateCreation:
    """Tests for agent state creation"""

    def test_orchestrator_creates_initial_state(self, agent_mode_text, sample_task):
        """Test TaskOrchestrator can create initial state"""
        agent = TaskOrchestrator(mode=agent_mode_text)
        state = agent.create_initial_state(sample_task)

        assert state.session_id == agent.session_id
        assert state.agent_type == agent.agent_type
        assert state.mode == agent.mode
        assert state.current_task == sample_task

    def test_executor_creates_initial_state(self, agent_mode_text, sample_task):
        """Test TaskExecutor can create initial state"""
        agent = TaskExecutor(mode=agent_mode_text)
        state = agent.create_initial_state(sample_task)

        assert state.session_id == agent.session_id
        assert state.agent_type == agent.agent_type
        assert state.current_task == sample_task

    def test_checker_creates_initial_state(self, agent_mode_text, sample_task):
        """Test TaskChecker can create initial state"""
        agent = TaskChecker(mode=agent_mode_text)
        state = agent.create_initial_state(sample_task)

        assert state.session_id == agent.session_id
        assert state.agent_type == agent.agent_type
        assert state.current_task == sample_task


@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentProcessInput:
    """Tests for agent process_input method"""

    async def test_orchestrator_process_input(self, agent_mode_text, sample_task):
        """Test TaskOrchestrator can process input"""
        agent = TaskOrchestrator(mode=agent_mode_text)

        result = await agent.process_input(sample_task)

        assert result is not None
        assert isinstance(result, dict)
        assert "status" in result
        assert "agent_type" in result
        assert result["agent_type"] == "task-orchestrator"

    async def test_executor_process_input(self, agent_mode_text, sample_task):
        """Test TaskExecutor can process input"""
        agent = TaskExecutor(mode=agent_mode_text)

        result = await agent.process_input(sample_task)

        assert result is not None
        assert isinstance(result, dict)
        assert "status" in result
        assert "agent_type" in result
        assert result["agent_type"] == "task-executor"

    async def test_checker_process_input(self, agent_mode_text, sample_task):
        """Test TaskChecker can process input"""
        agent = TaskChecker(mode=agent_mode_text)

        result = await agent.process_input(sample_task)

        assert result is not None
        assert isinstance(result, dict)
        assert "status" in result
        assert "agent_type" in result
        assert result["agent_type"] == "task-checker"

    async def test_agent_handles_string_input(self, agent_mode_text):
        """Test agents can handle string input"""
        agent = TaskOrchestrator(mode=agent_mode_text)
        result = await agent.process_input("Simple string task")

        assert result is not None
        assert isinstance(result, dict)

    async def test_agent_handles_dict_input(self, agent_mode_text):
        """Test agents can handle dictionary input"""
        agent = TaskOrchestrator(mode=agent_mode_text)
        result = await agent.process_input({"task": "Task from dict"})

        assert result is not None
        assert isinstance(result, dict)


@pytest.mark.integration
class TestAgentGraph:
    """Tests for agent graph construction"""

    def test_orchestrator_has_graph(self, agent_mode_text):
        """Test TaskOrchestrator has compiled graph"""
        agent = TaskOrchestrator(mode=agent_mode_text)

        assert hasattr(agent, 'graph')
        assert hasattr(agent, 'app')
        assert agent.app is not None

    def test_executor_has_graph(self, agent_mode_text):
        """Test TaskExecutor has compiled graph"""
        agent = TaskExecutor(mode=agent_mode_text)

        assert hasattr(agent, 'graph')
        assert hasattr(agent, 'app')
        assert agent.app is not None

    def test_checker_has_graph(self, agent_mode_text):
        """Test TaskChecker has compiled graph"""
        agent = TaskChecker(mode=agent_mode_text)

        assert hasattr(agent, 'graph')
        assert hasattr(agent, 'app')
        assert agent.app is not None
