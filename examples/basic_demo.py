#!/usr/bin/env python3
"""
Basic Demo - LangGraph Voice-Enabled Agent Framework

This example demonstrates the basic usage of the three main agents:
1. TaskOrchestrator - Strategic planning and task delegation
2. TaskExecutor - Implementation and development execution
3. TaskChecker - Quality assurance and validation

Usage:
    python examples/basic_demo.py

Requirements:
    - All dependencies installed (pip install -e .)
    - Environment variables configured (.env file)
    - Optional: MCP tool servers running for full functionality
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


async def demo_task_orchestrator():
    """Demonstrate TaskOrchestrator agent"""
    print("\n" + "="*70)
    print("Demo 1: Task Orchestrator - Strategic Planning")
    print("="*70)

    try:
        # Create orchestrator in text mode (no voice required)
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)

        logger.info(f"Created TaskOrchestrator: {orchestrator.agent_type}")
        logger.info(f"Session ID: {orchestrator.session_id}")
        logger.info(f"Mode: {orchestrator.mode.value}")

        # Create a test task
        task = "Build a REST API for user authentication with JWT tokens"

        logger.info(f"Processing task: {task}")

        # Process the task
        result = await orchestrator.process_input(task)

        logger.info(f"Task processing status: {result['status']}")
        logger.info(f"Agent type: {result['agent_type']}")

        print("\n✓ TaskOrchestrator demo completed successfully")
        return True

    except Exception as e:
        logger.error(f"TaskOrchestrator demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_task_executor():
    """Demonstrate TaskExecutor agent"""
    print("\n" + "="*70)
    print("Demo 2: Task Executor - Implementation & Execution")
    print("="*70)

    try:
        # Create executor in text mode
        executor = TaskExecutor(mode=AgentMode.TEXT)

        logger.info(f"Created TaskExecutor: {executor.agent_type}")
        logger.info(f"Session ID: {executor.session_id}")
        logger.info(f"Mode: {executor.mode.value}")

        # Create a test implementation task
        task = "Implement a function to validate email addresses using regex"

        logger.info(f"Processing task: {task}")

        # Process the task
        result = await executor.process_input(task)

        logger.info(f"Task processing status: {result['status']}")
        logger.info(f"Agent type: {result['agent_type']}")

        print("\n✓ TaskExecutor demo completed successfully")
        return True

    except Exception as e:
        logger.error(f"TaskExecutor demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_task_checker():
    """Demonstrate TaskChecker agent"""
    print("\n" + "="*70)
    print("Demo 3: Task Checker - Quality Assurance & Validation")
    print("="*70)

    try:
        # Create checker in text mode
        checker = TaskChecker(mode=AgentMode.TEXT)

        logger.info(f"Created TaskChecker: {checker.agent_type}")
        logger.info(f"Session ID: {checker.session_id}")
        logger.info(f"Mode: {checker.mode.value}")

        # Create a test validation task
        task = "Validate the email validation function with comprehensive test cases"

        logger.info(f"Processing task: {task}")

        # Process the task
        result = await checker.process_input(task)

        logger.info(f"Task processing status: {result['status']}")
        logger.info(f"Agent type: {result['agent_type']}")

        print("\n✓ TaskChecker demo completed successfully")
        return True

    except Exception as e:
        logger.error(f"TaskChecker demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_agent_collaboration():
    """Demonstrate agent collaboration workflow"""
    print("\n" + "="*70)
    print("Demo 4: Agent Collaboration - Complete Workflow")
    print("="*70)

    try:
        # Create all three agents
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        executor = TaskExecutor(mode=AgentMode.TEXT)
        checker = TaskChecker(mode=AgentMode.TEXT)

        logger.info("Created all three agents for collaboration demo")

        # Define a complex task
        task = "Create a secure password hashing system with bcrypt"

        # Step 1: Orchestrator plans the task
        logger.info("Step 1: Orchestrator planning...")
        orchestrator_result = await orchestrator.process_input(task)

        # Step 2: Executor implements the task
        logger.info("Step 2: Executor implementing...")
        executor_result = await executor.process_input(task)

        # Step 3: Checker validates the implementation
        logger.info("Step 3: Checker validating...")
        checker_result = await checker.process_input(f"Validate: {task}")

        print("\n✓ Agent collaboration demo completed successfully")
        print(f"  - Orchestration: {orchestrator_result['status']}")
        print(f"  - Execution: {executor_result['status']}")
        print(f"  - Validation: {checker_result['status']}")

        return True

    except Exception as e:
        logger.error(f"Agent collaboration demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("LangGraph Voice-Enabled Agent Framework - Basic Demo")
    print("="*70)

    results = []

    # Run individual agent demos
    results.append(("TaskOrchestrator", await demo_task_orchestrator()))
    results.append(("TaskExecutor", await demo_task_executor()))
    results.append(("TaskChecker", await demo_task_checker()))

    # Run collaboration demo
    results.append(("Agent Collaboration", await demo_agent_collaboration()))

    # Summary
    print("\n" + "="*70)
    print("Demo Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for demo_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {demo_name}")

    print(f"\nTotal: {passed}/{total} demos completed successfully")

    if passed == total:
        print("\n🎉 All demos passed! Framework is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} demo(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
