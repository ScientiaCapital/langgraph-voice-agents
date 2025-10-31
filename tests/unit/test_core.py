"""
Unit tests for core module components
"""

import pytest
from core import AgentState, AgentMode, BaseAgent


class TestAgentState:
    """Tests for AgentState dataclass"""

    def test_agent_state_creation(self):
        """Test basic agent state creation"""
        state = AgentState(
            session_id="test-session",
            agent_type="test-agent",
            mode=AgentMode.TEXT,
            current_task="Test task"
        )

        assert state.session_id == "test-session"
        assert state.agent_type == "test-agent"
        assert state.mode == AgentMode.TEXT
        assert state.current_task == "Test task"

    def test_agent_state_defaults(self):
        """Test agent state default values"""
        state = AgentState(
            session_id="test",
            agent_type="test",
            mode=AgentMode.TEXT
        )

        assert state.task_history == []
        assert state.task_context == {}
        assert state.messages == []
        assert state.errors == []
        assert state.retry_count == 0

    def test_agent_state_mutable_fields(self):
        """Test that mutable fields are properly initialized"""
        state1 = AgentState(session_id="1", agent_type="test", mode=AgentMode.TEXT)
        state2 = AgentState(session_id="2", agent_type="test", mode=AgentMode.TEXT)

        state1.task_history.append("task1")
        state2.task_history.append("task2")

        # Ensure they don't share the same list
        assert len(state1.task_history) == 1
        assert len(state2.task_history) == 1
        assert state1.task_history[0] == "task1"
        assert state2.task_history[0] == "task2"


class TestAgentMode:
    """Tests for AgentMode enum"""

    def test_agent_mode_values(self):
        """Test agent mode enum values"""
        assert AgentMode.TEXT.value == "text"
        assert AgentMode.VOICE.value == "voice"
        assert AgentMode.HYBRID.value == "hybrid"

    def test_agent_mode_comparison(self):
        """Test agent mode comparison"""
        mode1 = AgentMode.TEXT
        mode2 = AgentMode.TEXT
        mode3 = AgentMode.VOICE

        assert mode1 == mode2
        assert mode1 != mode3


class TestBaseAgent:
    """Tests for BaseAgent abstract class"""

    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAgent(agent_type="test", mode=AgentMode.TEXT)

    def test_base_agent_requires_build_graph(self):
        """Test that _build_graph is required"""
        # This is implicitly tested by the above, but we can verify
        # that the abstract method exists
        assert hasattr(BaseAgent, '_build_graph')
        assert hasattr(BaseAgent, 'process_input')
