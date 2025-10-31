#!/usr/bin/env python3
"""
Error Handling Examples - LangGraph Voice Agent Framework

This example demonstrates proper error handling patterns when using agents.

Usage:
    python examples/error_handling/error_handling_demo.py
"""

import asyncio
import logging
from typing import Optional

from core import AgentMode
from agents import TaskOrchestrator, TaskExecutor, TaskChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_1_basic_error_handling():
    """Example 1: Basic try/except error handling"""
    print("\n" + "="*70)
    print("Example 1: Basic Error Handling")
    print("="*70)

    try:
        agent = TaskOrchestrator(mode=AgentMode.TEXT)
        result = await agent.process_input("Design a REST API")

        logger.info(f"Success: {result['status']}")
        return True

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False


async def example_2_handling_invalid_input():
    """Example 2: Handling invalid input gracefully"""
    print("\n" + "="*70)
    print("Example 2: Invalid Input Handling")
    print("="*70)

    agent = TaskOrchestrator(mode=AgentMode.TEXT)

    # Test various invalid inputs
    invalid_inputs = [
        ("", "Empty string"),
        (None, "None value"),
        ({}, "Empty dict"),
        ({"no_task_key": "value"}, "Dict without 'task' key"),
    ]

    for invalid_input, description in invalid_inputs:
        try:
            logger.info(f"Testing: {description}")
            result = await agent.process_input(invalid_input or {})
            logger.info(f"Result: {result['status']}")

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Expected error for {description}: {e}")

        except Exception as e:
            logger.error(f"Unexpected error for {description}: {e}")


async def example_3_agent_initialization_errors():
    """Example 3: Handling agent initialization errors"""
    print("\n" + "="*70)
    print("Example 3: Agent Initialization Error Handling")
    print("="*70)

    try:
        # Attempt to create agent with invalid configuration
        agent = TaskOrchestrator(
            mode=AgentMode.TEXT,
            checkpointer_path="/invalid/path/to/db.sqlite"
        )
        logger.info("Agent created successfully (or with fallback)")

    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        logger.info("Consider using default :memory: path")

    except Exception as e:
        logger.error(f"Initialization error: {e}")


async def example_4_timeout_handling():
    """Example 4: Handling long-running operations with timeouts"""
    print("\n" + "="*70)
    print("Example 4: Timeout Handling")
    print("="*70)

    agent = TaskExecutor(mode=AgentMode.TEXT)

    try:
        # Set a timeout for the operation
        result = await asyncio.wait_for(
            agent.process_input("Complex task that might take long"),
            timeout=30.0  # 30 seconds
        )

        logger.info(f"Completed within timeout: {result['status']}")

    except asyncio.TimeoutError:
        logger.error("Operation timed out after 30 seconds")
        logger.info("Consider breaking task into smaller chunks")

    except Exception as e:
        logger.error(f"Error: {e}")


async def example_5_retry_logic():
    """Example 5: Implementing retry logic for transient errors"""
    print("\n" + "="*70)
    print("Example 5: Retry Logic")
    print("="*70)

    agent = TaskOrchestrator(mode=AgentMode.TEXT)
    max_retries = 3
    retry_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")

            result = await agent.process_input("Design microservices architecture")

            logger.info(f"Success on attempt {attempt + 1}")
            return result

        except ConnectionError as e:
            logger.warning(f"Connection error: {e}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Max retries reached, giving up")
                raise

        except Exception as e:
            logger.error(f"Non-retryable error: {e}")
            raise


async def example_6_graceful_degradation():
    """Example 6: Graceful degradation when optional features fail"""
    print("\n" + "="*70)
    print("Example 6: Graceful Degradation")
    print("="*70)

    try:
        # Try to create voice-enabled agent
        # (will fail if LiveKit not configured, but that's OK)
        from voice import LiveKitConfig

        # This will work in TEXT mode even if voice fails
        agent = TaskOrchestrator(mode=AgentMode.TEXT)

        logger.info("Agent created in TEXT mode")
        logger.info("Voice features not available, but agent still functional")

        result = await agent.process_input("Plan authentication system")
        logger.info(f"Result: {result['status']}")

    except ImportError as e:
        logger.warning(f"Voice features not available: {e}")
        logger.info("Falling back to TEXT mode only")

        agent = TaskOrchestrator(mode=AgentMode.TEXT)
        result = await agent.process_input("Plan authentication system")
        logger.info(f"Completed in TEXT mode: {result['status']}")


async def example_7_context_manager_pattern():
    """Example 7: Using context managers for resource cleanup"""
    print("\n" + "="*70)
    print("Example 7: Context Manager Pattern")
    print("="*70)

    class AgentSession:
        """Context manager for agent lifecycle"""

        def __init__(self, agent_class, mode=AgentMode.TEXT):
            self.agent_class = agent_class
            self.mode = mode
            self.agent = None

        async def __aenter__(self):
            logger.info("Initializing agent session...")
            self.agent = self.agent_class(mode=self.mode)
            return self.agent

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            logger.info("Cleaning up agent session...")
            # Perform any cleanup (close connections, save state, etc.)

            if exc_type:
                logger.error(f"Session ended with error: {exc_val}")

            return False  # Don't suppress exceptions

    # Use the context manager
    async with AgentSession(TaskOrchestrator) as agent:
        result = await agent.process_input("Design API gateway")
        logger.info(f"Result: {result['status']}")

    logger.info("Session automatically cleaned up")


async def example_8_error_state_tracking():
    """Example 8: Tracking errors in agent state"""
    print("\n" + "="*70)
    print("Example 8: Error State Tracking")
    print("="*70)

    agent = TaskOrchestrator(mode=AgentMode.TEXT)

    try:
        # Create initial state to track errors
        state = agent.create_initial_state("Complex task")

        logger.info(f"Initial error count: {len(state.errors)}")
        logger.info(f"Initial retry count: {state.retry_count}")

        # Process task
        result = await agent.process_input("Design distributed system")

        logger.info(f"Task completed: {result['status']}")
        # Note: In production, you'd access the actual state from the graph

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        # Errors would be tracked in the agent state


async def example_9_multiple_agent_error_coordination():
    """Example 9: Error handling across multiple agents"""
    print("\n" + "="*70)
    print("Example 9: Multi-Agent Error Coordination")
    print("="*70)

    task = "Build authentication system"
    results = {"orchestrator": None, "executor": None, "checker": None}
    errors = []

    # Try orchestration
    try:
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        results["orchestrator"] = await orchestrator.process_input(task)
        logger.info("✓ Orchestration completed")

    except Exception as e:
        errors.append(("orchestrator", e))
        logger.error(f"✗ Orchestration failed: {e}")

    # Try execution (even if orchestration failed)
    try:
        executor = TaskExecutor(mode=AgentMode.TEXT)
        results["executor"] = await executor.process_input(task)
        logger.info("✓ Execution completed")

    except Exception as e:
        errors.append(("executor", e))
        logger.error(f"✗ Execution failed: {e}")

    # Try validation (only if execution succeeded)
    if results["executor"]:
        try:
            checker = TaskChecker(mode=AgentMode.TEXT)
            results["checker"] = await checker.process_input(f"Validate: {task}")
            logger.info("✓ Validation completed")

        except Exception as e:
            errors.append(("checker", e))
            logger.error(f"✗ Validation failed: {e}")

    # Summary
    successes = sum(1 for r in results.values() if r is not None)
    logger.info(f"\nSummary: {successes}/3 agents completed successfully")

    if errors:
        logger.warning(f"Errors encountered: {len(errors)}")
        for agent_name, error in errors:
            logger.warning(f"  - {agent_name}: {error}")


async def example_10_logging_and_debugging():
    """Example 10: Enhanced logging for debugging"""
    print("\n" + "="*70)
    print("Example 10: Logging and Debugging")
    print("="*70)

    # Set debug logging level
    logging.getLogger().setLevel(logging.DEBUG)

    try:
        agent = TaskOrchestrator(mode=AgentMode.TEXT)

        logger.debug(f"Agent type: {agent.agent_type}")
        logger.debug(f"Agent mode: {agent.mode}")
        logger.debug(f"Session ID: {agent.session_id}")

        result = await agent.process_input("Design caching layer")

        logger.debug(f"Result keys: {result.keys()}")
        logger.info(f"Final status: {result['status']}")

    except Exception as e:
        logger.exception("Full exception traceback:")

    finally:
        # Reset logging level
        logging.getLogger().setLevel(logging.INFO)


async def main():
    """Run all error handling examples"""
    print("\n" + "="*70)
    print("LangGraph Voice Agent Framework - Error Handling Examples")
    print("="*70)

    examples = [
        ("Basic Error Handling", example_1_basic_error_handling),
        ("Invalid Input Handling", example_2_handling_invalid_input),
        ("Initialization Errors", example_3_agent_initialization_errors),
        ("Timeout Handling", example_4_timeout_handling),
        ("Retry Logic", example_5_retry_logic),
        ("Graceful Degradation", example_6_graceful_degradation),
        ("Context Manager Pattern", example_7_context_manager_pattern),
        ("Error State Tracking", example_8_error_state_tracking),
        ("Multi-Agent Coordination", example_9_multiple_agent_error_coordination),
        ("Logging and Debugging", example_10_logging_and_debugging),
    ]

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}")

    print("\n" + "="*70)
    print("Error Handling Examples Complete")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
