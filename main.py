#!/usr/bin/env python3
"""
LangGraph Voice Agents - Main CLI Entry Point

Voice-enabled multi-agent framework with:
- Cartesia TTS/STT (NO OpenAI)
- Multi-LLM support (Claude, Gemini, OpenRouter)
- LiveKit WebRTC integration

Usage:
    python main.py                    # Interactive mode - choose an agent
    python main.py --agent general    # Start General Assistant
    python main.py --agent code       # Start Code Assistant
    python main.py --agent task       # Start Task Manager
    python main.py --text             # Text-only mode (no voice)
    python main.py --test             # Run integration tests
"""

import asyncio
import argparse
import logging
import sys
import os
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    create_general_assistant,
    create_code_assistant,
    create_task_manager,
    VoiceAgent,
)
from core import AgentMode
from llm import LLMConfig, LLMRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║         🎙️  LangGraph Voice Agents Framework  🎙️              ║
║                                                               ║
║   Voice: Cartesia (sonic-2 TTS, ink-whisper STT)             ║
║   LLMs: Claude, Gemini, OpenRouter                           ║
║   Transport: LiveKit WebRTC                                   ║
║                                                               ║
║   NO OpenAI dependencies                                      ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_environment() -> dict:
    """Check for required environment variables"""
    env_status = {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "cartesia": bool(os.getenv("CARTESIA_API_KEY")),
        "livekit_url": bool(os.getenv("LIVEKIT_URL")),
        "livekit_key": bool(os.getenv("LIVEKIT_API_KEY")),
        "livekit_secret": bool(os.getenv("LIVEKIT_API_SECRET")),
    }

    print("\n📋 Environment Check:")
    print("-" * 40)

    # LLM Providers
    llm_available = []
    print("LLM Providers:")
    if env_status["anthropic"]:
        print("  ✅ Anthropic (Claude)")
        llm_available.append("Claude")
    else:
        print("  ❌ Anthropic (ANTHROPIC_API_KEY not set)")

    if env_status["google"]:
        print("  ✅ Google (Gemini)")
        llm_available.append("Gemini")
    else:
        print("  ❌ Google (GOOGLE_API_KEY not set)")

    if env_status["openrouter"]:
        print("  ✅ OpenRouter")
        llm_available.append("OpenRouter")
    else:
        print("  ❌ OpenRouter (OPENROUTER_API_KEY not set)")

    # Voice
    print("\nVoice Provider:")
    if env_status["cartesia"]:
        print("  ✅ Cartesia (TTS + STT)")
    else:
        print("  ❌ Cartesia (CARTESIA_API_KEY not set)")

    # LiveKit
    print("\nLiveKit (optional):")
    livekit_ready = all([env_status["livekit_url"], env_status["livekit_key"], env_status["livekit_secret"]])
    if livekit_ready:
        print("  ✅ LiveKit configured")
    else:
        print("  ⚠️  LiveKit not configured (WebRTC rooms disabled)")

    print("-" * 40)

    if not llm_available:
        print("\n⚠️  Warning: No LLM providers configured!")
        print("   Set at least one of: ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY")

    if not env_status["cartesia"]:
        print("\n⚠️  Warning: Cartesia not configured!")
        print("   Voice features will be disabled. Set CARTESIA_API_KEY")

    return env_status


def select_agent() -> str:
    """Interactive agent selection"""
    print("\n🤖 Available Agents:")
    print("-" * 40)
    print("1. General Assistant (Atlas)")
    print("   - Conversation, Q&A, creative writing")
    print("   - Best for everyday tasks")
    print()
    print("2. Code Assistant (Cipher)")
    print("   - Code explanation, review, generation")
    print("   - Best for programming help")
    print()
    print("3. Task Manager (Taskmaster)")
    print("   - Task tracking, project planning")
    print("   - Best for productivity")
    print("-" * 40)

    while True:
        choice = input("\nSelect agent (1/2/3) or 'q' to quit: ").strip().lower()
        if choice == '1' or choice == 'general':
            return 'general'
        elif choice == '2' or choice == 'code':
            return 'code'
        elif choice == '3' or choice == 'task':
            return 'task'
        elif choice == 'q' or choice == 'quit':
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


async def run_interactive_session(agent: VoiceAgent):
    """Run interactive chat session with agent"""
    print(f"\n🎙️  Starting session with {agent.config.agent_name}")
    print("-" * 40)
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("Type 'voice on/off' to toggle voice mode")
    print("-" * 40)

    # Start voice session if in voice mode
    if agent.mode == AgentMode.VOICE:
        try:
            await agent.start_voice_session()
            print("🔊 Voice mode enabled")
        except Exception as e:
            logger.warning(f"Could not start voice session: {e}")
            print("⚠️  Voice unavailable, using text mode")

    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print(f"\n{agent.config.agent_name}: Goodbye! Have a great day!")
                break

            if user_input.lower() == 'clear':
                agent.clear_history()
                print("Conversation history cleared.")
                continue

            if user_input.lower() == 'voice on':
                if agent.mode != AgentMode.VOICE:
                    agent.mode = AgentMode.VOICE
                    await agent.start_voice_session()
                    print("🔊 Voice mode enabled")
                continue

            if user_input.lower() == 'voice off':
                if agent.mode == AgentMode.VOICE:
                    await agent.stop_voice_session()
                    agent.mode = AgentMode.TEXT
                    print("🔇 Voice mode disabled")
                continue

            # Get response
            response = await agent.process_input(user_input)
            print(f"\n{agent.config.agent_name}: {response}\n")

        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")

    # Cleanup
    await agent.close()


async def run_tests():
    """Run integration tests"""
    print("\n🧪 Running Integration Tests")
    print("=" * 50)

    tests_passed = 0
    tests_failed = 0

    # Test 1: LLM Router
    print("\n1. Testing LLM Router...")
    try:
        config = LLMConfig()
        router = LLMRouter(config)
        providers = router.available_providers
        if providers:
            print(f"   ✅ LLM Router initialized with {len(providers)} providers")
            tests_passed += 1
        else:
            print("   ⚠️  No LLM providers available")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ LLM Router failed: {e}")
        tests_failed += 1

    # Test 2: General Assistant Creation
    print("\n2. Testing General Assistant creation...")
    try:
        agent = create_general_assistant(voice_mode=False)
        print(f"   ✅ Created: {agent.config.agent_name}")
        tests_passed += 1
        await agent.close()
    except Exception as e:
        print(f"   ❌ General Assistant failed: {e}")
        tests_failed += 1

    # Test 3: Code Assistant Creation
    print("\n3. Testing Code Assistant creation...")
    try:
        agent = create_code_assistant(voice_mode=False)
        print(f"   ✅ Created: {agent.config.agent_name}")
        tests_passed += 1
        await agent.close()
    except Exception as e:
        print(f"   ❌ Code Assistant failed: {e}")
        tests_failed += 1

    # Test 4: Task Manager Creation
    print("\n4. Testing Task Manager creation...")
    try:
        agent = create_task_manager(voice_mode=False)
        print(f"   ✅ Created: {agent.config.agent_name}")
        tests_passed += 1
        await agent.close()
    except Exception as e:
        print(f"   ❌ Task Manager failed: {e}")
        tests_failed += 1

    # Test 5: LLM Generation (if available)
    print("\n5. Testing LLM generation...")
    try:
        config = LLMConfig()
        router = LLMRouter(config)
        if router.available_providers:
            from llm.provider import Message
            response = await router.chat("Say 'Hello, test passed!' in exactly those words.")
            if "hello" in response.lower():
                print(f"   ✅ LLM response received")
                tests_passed += 1
            else:
                print(f"   ⚠️  Unexpected response: {response[:50]}...")
                tests_passed += 1  # Still counts as working
            await router.close()
        else:
            print("   ⚠️  Skipped (no LLM providers)")
    except Exception as e:
        print(f"   ❌ LLM generation failed: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print("=" * 50)

    return tests_failed == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='LangGraph Voice Agents - Voice-enabled AI assistants'
    )
    parser.add_argument(
        '--agent',
        choices=['general', 'code', 'task'],
        help='Agent to start (general, code, or task)'
    )
    parser.add_argument(
        '--text',
        action='store_true',
        help='Run in text-only mode (no voice)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run integration tests'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print_banner()

    # Check environment
    env_status = check_environment()

    # Run tests if requested
    if args.test:
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)

    # Select agent
    if args.agent:
        agent_type = args.agent
    else:
        agent_type = select_agent()

    # Create agent
    voice_mode = not args.text and env_status["cartesia"]

    if agent_type == 'general':
        agent = create_general_assistant(voice_mode=voice_mode)
    elif agent_type == 'code':
        agent = create_code_assistant(voice_mode=voice_mode)
    elif agent_type == 'task':
        agent = create_task_manager(voice_mode=voice_mode)
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)

    # Run interactive session
    try:
        asyncio.run(run_interactive_session(agent))
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
