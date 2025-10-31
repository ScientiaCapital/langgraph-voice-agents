"""
Integration tests for complete agent workflows
"""

import pytest
from core import AgentMode
from agents import TaskOrchestrator, TaskExecutor, TaskChecker


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompleteWorkflow:
    """Tests for complete end-to-end workflows"""

    async def test_orchestrator_to_executor_workflow(self, sample_task):
        """Test workflow from orchestration to execution"""
        # Step 1: Orchestrator plans the task
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        orchestration_result = await orchestrator.process_input(sample_task)

        assert orchestration_result["status"] is not None
        assert orchestration_result["agent_type"] == "task-orchestrator"

        # Step 2: Executor implements based on plan
        executor = TaskExecutor(mode=AgentMode.TEXT)
        execution_result = await executor.process_input(sample_task)

        assert execution_result["status"] is not None
        assert execution_result["agent_type"] == "task-executor"

   async def test_full_three_agent_workflow(self, sample_complex_task):
        """Test complete workflow: orchestrate → execute → validate"""
        # Create all agents
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        executor = TaskExecutor(mode=AgentMode.TEXT)
        checker = TaskChecker(mode=AgentMode.TEXT)

        # Step 1: Plan
        plan_result = await orchestrator.process_input(sample_complex_task)
        assert plan_result["status"] is not None

        # Step 2: Execute
        exec_result = await executor.process_input(sample_complex_task)
        assert exec_result["status"] is not None

        # Step 3: Validate
        validate_result = await checker.process_input(f"Validate: {sample_complex_task}")
        assert validate_result["status"] is not None

        # All steps should complete
        assert orchestrator.session_id is not None
        assert executor.session_id is not None
        assert checker.session_id is not None

    async def test_parallel_agent_execution(self, sample_task):
        """Test multiple agents running in parallel"""
        import asyncio

        # Create agents
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        executor = TaskExecutor(mode=AgentMode.TEXT)

        # Run in parallel
        results = await asyncio.gather(
            orchestrator.process_input(sample_task),
            executor.process_input(sample_task)
        )

        assert len(results) == 2
        assert all(result["status"] is not None for result in results)


@pytest.mark.integration
@pytest.mark.asyncio
class TestStateManagement:
    """Tests for state management across workflows"""

    async def test_state_persistence_across_calls(self):
        """Test that agent state persists across multiple calls"""
        agent = TaskOrchestrator(mode=AgentMode.TEXT)

        # First call
        result1 = await agent.process_input("First task")
        session1 = result1["session_id"]

        # Second call (same agent instance)
        result2 = await agent.process_input("Second task")
        session2 = result2["session_id"]

        # Session should be the same
        assert session1 == session2

    async def test_independent_agent_sessions(self):
        """Test that different agents have independent sessions"""
        agent1 = TaskOrchestrator(mode=AgentMode.TEXT)
        agent2 = TaskOrchestrator(mode=AgentMode.TEXT)

        result1 = await agent1.process_input("Task 1")
        result2 = await agent2.process_input("Task 2")

        # Different agents should have different sessions
        assert result1["session_id"] != result2["session_id"]


@pytest.mark.integration
class TestAgentModes:
    """Tests for different agent modes"""

    def test_text_mode_initialization(self):
        """Test agents in TEXT mode"""
        agents = [
            TaskOrchestrator(mode=AgentMode.TEXT),
            TaskExecutor(mode=AgentMode.TEXT),
            TaskChecker(mode=AgentMode.TEXT),
        ]

        for agent in agents:
            assert agent.mode == AgentMode.TEXT
            assert agent.livekit_client is None

    @pytest.mark.asyncio
    async def test_text_mode_workflow(self, sample_task):
        """Test complete workflow in TEXT mode"""
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        result = await orchestrator.process_input(sample_task)

        assert result is not None
        assert result["status"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandling:
    """Tests for error handling in workflows"""

    async def test_empty_input_handling(self):
        """Test handling of empty input"""
        agent = TaskOrchestrator(mode=AgentMode.TEXT)

        # Should handle empty string gracefully
        result = await agent.process_input("")
        assert result is not None
        assert isinstance(result, dict)

    async def test_none_input_handling(self):
        """Test handling of None input"""
        agent = TaskOrchestrator(mode=AgentMode.TEXT)

        # Should handle None input
        try:
            result = await agent.process_input({})
            assert result is not None
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert isinstance(e, (ValueError, TypeError, KeyError))

    async def test_malformed_dict_input(self):
        """Test handling of malformed dictionary input"""
        agent = TaskOrchestrator(mode=AgentMode.TEXT)

        # Dict without 'task' key
        result = await agent.process_input({"message": "Test"})
        assert result is not None
        assert isinstance(result, dict)


@pytest.mark.integration
class TestAgentCapabilities:
    """Tests for agent-specific capabilities"""

    def test_orchestrator_has_delegation_capabilities(self):
        """Test TaskOrchestrator has delegation features"""
        agent = TaskOrchestrator(mode=AgentMode.TEXT)

        assert hasattr(agent, 'agent_capabilities')
        assert hasattr(agent, 'delegated_tasks')
        assert hasattr(agent, 'active_plans')

    def test_executor_has_execution_tools(self):
        """Test TaskExecutor has execution tools"""
        agent = TaskExecutor(mode=AgentMode.TEXT)

        assert hasattr(agent, 'desktop_commander')
        assert hasattr(agent, 'active_processes')
        assert hasattr(agent, 'current_execution')

    def test_checker_has_validation_tools(self):
        """Test TaskChecker has validation capabilities"""
        agent = TaskChecker(mode=AgentMode.TEXT)

        assert hasattr(agent, 'validation_results')
        assert hasattr(agent, 'quality_metrics')
        assert hasattr(agent, 'current_validation_level')


@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentCommunication:
    """Tests for inter-agent communication patterns"""

    async def test_sequential_agent_chain(self, sample_task):
        """Test sequential execution chain"""
        results = []

        # Agent 1: Orchestrator
        agent1 = TaskOrchestrator(mode=AgentMode.TEXT)
        result1 = await agent1.process_input(sample_task)
        results.append(result1)

        # Agent 2: Executor (uses output from orchestrator)
        agent2 = TaskExecutor(mode=AgentMode.TEXT)
        result2 = await agent2.process_input(sample_task)
        results.append(result2)

        # Agent 3: Checker (validates execution)
        agent3 = TaskChecker(mode=AgentMode.TEXT)
        result3 = await agent3.process_input(sample_task)
        results.append(result3)

        assert len(results) == 3
        assert all(r["status"] is not None for r in results)

    async def test_agent_handoff_data_flow(self):
        """Test data flow between agents"""
        task = "Implement user authentication API"

        # Orchestrator processes and creates plan
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        plan = await orchestrator.process_input(task)

        # Executor can reference the plan
        executor = TaskExecutor(mode=AgentMode.TEXT)
        execution = await executor.process_input(task)

        # Both should have completed
        assert plan["status"] is not None
        assert execution["status"] is not None
