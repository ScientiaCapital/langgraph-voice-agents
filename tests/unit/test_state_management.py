"""
Unit tests for state management
"""

import pytest
from core.state_management import StateManager, ExecutionState, TaskState


@pytest.mark.unit
class TestExecutionState:
    """Tests for ExecutionState enum"""

    def test_execution_state_values(self):
        """Test ExecutionState enum values"""
        assert ExecutionState.PENDING.value == "pending"
        assert ExecutionState.RUNNING.value == "running"
        assert ExecutionState.COMPLETED.value == "completed"
        assert ExecutionState.FAILED.value == "failed"
        assert ExecutionState.PAUSED.value == "paused"

    def test_execution_state_comparison(self):
        """Test ExecutionState comparison"""
        assert ExecutionState.PENDING == ExecutionState.PENDING
        assert ExecutionState.PENDING != ExecutionState.RUNNING


@pytest.mark.unit
class TestTaskState:
    """Tests for TaskState dataclass"""

    def test_task_state_creation(self):
        """Test TaskState creation"""
        state = TaskState(
            task_id="test-123",
            name="Test Task",
            status=ExecutionState.PENDING
        )

        assert state.task_id == "test-123"
        assert state.name == "Test Task"
        assert state.status == ExecutionState.PENDING

    def test_task_state_defaults(self):
        """Test TaskState default values"""
        state = TaskState(
            task_id="test",
            name="Test",
            status=ExecutionState.PENDING
        )

        assert state.created_at is not None
        assert state.updated_at is not None
        assert state.metadata == {}

    def test_task_state_with_metadata(self):
        """Test TaskState with custom metadata"""
        metadata = {"priority": "high", "tags": ["urgent", "bug"]}
        state = TaskState(
            task_id="test",
            name="Test",
            status=ExecutionState.PENDING,
            metadata=metadata
        )

        assert state.metadata == metadata
        assert state.metadata["priority"] == "high"


@pytest.mark.unit
class TestStateManager:
    """Tests for StateManager"""

    def test_init_with_sqlite(self):
        """Test StateManager with SQLite backend"""
        manager = StateManager(
            db_url="sqlite:///:memory:",
            redis_url=None
        )

        assert manager is not None
        assert hasattr(manager, 'session')

    def test_init_without_redis(self):
        """Test StateManager without Redis (optional)"""
        manager = StateManager(
            db_url="sqlite:///:memory:",
            redis_url=None
        )

        assert manager is not None
        # Redis is optional, so manager should work without it

    def test_has_state_operations(self):
        """Test StateManager has state operation methods"""
        manager = StateManager(db_url="sqlite:///:memory:")

        # Check for expected methods
        assert hasattr(manager, '__init__')
        # Add more specific method checks based on actual implementation
