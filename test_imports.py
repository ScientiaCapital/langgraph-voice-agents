#!/usr/bin/env python3
"""
Smoke test to verify all critical imports and basic agent instantiation.
Run this after fixing structural issues to ensure the framework loads correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_core_imports():
    """Test core module imports"""
    print("Testing core imports...")
    try:
        from core import (
            BaseAgent,
            AgentState,
            AgentMode,
            MultiModalMixin,
            ErrorHandlingMixin,
            StateManager,
            ExecutionState,
            TaskState,
        )
        print("✓ Core imports successful")
        return True
    except Exception as e:
        print(f"✗ Core imports failed: {e}")
        return False


def test_tool_imports():
    """Test MCP tool adapter imports"""
    print("\nTesting tool adapter imports...")
    try:
        from tools import (
            SequentialThinkingAdapter,
            SerenaAdapter,
            Context7Adapter,
            TaskMasterAdapter,
            ShrimpTaskManagerAdapter,
            DesktopCommanderAdapter,
        )
        print("✓ Tool adapter imports successful")
        return True
    except Exception as e:
        print(f"✗ Tool adapter imports failed: {e}")
        return False


def test_voice_imports():
    """Test voice integration imports"""
    print("\nTesting voice integration imports...")
    try:
        from voice import LiveKitClient, LiveKitConfig, AudioProcessor
        print("✓ Voice integration imports successful")
        return True
    except Exception as e:
        print(f"✗ Voice integration imports failed: {e}")
        return False


def test_agent_imports():
    """Test agent imports"""
    print("\nTesting agent imports...")
    try:
        from agents import TaskOrchestrator, TaskExecutor, TaskChecker
        print("✓ Agent imports successful")
        return True
    except Exception as e:
        print(f"✗ Agent imports failed: {e}")
        return False


def test_agent_instantiation():
    """Test basic agent instantiation"""
    print("\nTesting agent instantiation...")

    try:
        from agents import TaskOrchestrator, TaskExecutor, TaskChecker
        from core import AgentMode

        # Test TaskOrchestrator
        print("  Creating TaskOrchestrator...")
        orchestrator = TaskOrchestrator(mode=AgentMode.TEXT)
        print(f"  ✓ TaskOrchestrator created (type: {orchestrator.agent_type})")

        # Test TaskExecutor
        print("  Creating TaskExecutor...")
        executor = TaskExecutor(mode=AgentMode.TEXT)
        print(f"  ✓ TaskExecutor created (type: {executor.agent_type})")

        # Test TaskChecker
        print("  Creating TaskChecker...")
        checker = TaskChecker(mode=AgentMode.TEXT)
        print(f"  ✓ TaskChecker created (type: {checker.agent_type})")

        print("✓ Agent instantiation successful")
        return True

    except Exception as e:
        print(f"✗ Agent instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_state_creation():
    """Test agent state creation"""
    print("\nTesting agent state creation...")
    try:
        from agents import TaskOrchestrator
        from core import AgentMode

        agent = TaskOrchestrator(mode=AgentMode.TEXT)
        state = agent.create_initial_state("Test task")

        print(f"  Session ID: {state.session_id}")
        print(f"  Agent Type: {state.agent_type}")
        print(f"  Mode: {state.mode}")
        print(f"  Current Task: {state.current_task}")
        print("✓ Agent state creation successful")
        return True

    except Exception as e:
        print(f"✗ Agent state creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests"""
    print("=" * 60)
    print("LangGraph Voice Agent Framework - Smoke Test")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Core Imports", test_core_imports()))
    results.append(("Tool Imports", test_tool_imports()))
    results.append(("Voice Imports", test_voice_imports()))
    results.append(("Agent Imports", test_agent_imports()))
    results.append(("Agent Instantiation", test_agent_instantiation()))
    results.append(("Agent State Creation", test_agent_state_creation()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All smoke tests passed! Framework is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
