#!/usr/bin/env python3
"""
Voice Demo - LiveKit Integration

This example demonstrates voice-enabled agent interactions using LiveKit.

Requirements:
    - LiveKit server running and accessible
    - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env
    - OpenAI API key for Whisper (STT) and TTS

Usage:
    python examples/voice_demo.py

Note: This is a more advanced example that requires proper LiveKit setup.
"""

import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from core import AgentMode
from agents import TaskOrchestrator
from voice import LiveKitConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_voice_requirements() -> bool:
    """Check if all required environment variables are set"""
    required_vars = [
        'LIVEKIT_URL',
        'LIVEKIT_API_KEY',
        'LIVEKIT_API_SECRET',
        'OPENAI_API_KEY'
    ]

    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please configure your .env file based on .env.example")
        return False

    return True


async def demo_voice_agent():
    """Demonstrate voice-enabled agent"""
    print("\n" + "="*70)
    print("Voice Demo - LiveKit Integration")
    print("="*70)

    # Check requirements
    if not check_voice_requirements():
        return False

    try:
        # Configure LiveKit
        livekit_config = LiveKitConfig(
            url=os.getenv('LIVEKIT_URL'),
            api_key=os.getenv('LIVEKIT_API_KEY'),
            api_secret=os.getenv('LIVEKIT_API_SECRET'),
            room_name=os.getenv('LIVEKIT_ROOM_NAME', 'agent-demo')
        )

        # Create voice-enabled orchestrator
        orchestrator = TaskOrchestrator(
            mode=AgentMode.VOICE,
            livekit_config=livekit_config
        )

        logger.info(f"Created voice-enabled TaskOrchestrator")
        logger.info(f"LiveKit room: {livekit_config.room_name}")

        # Test task
        task = "Plan a microservices architecture for an e-commerce platform"

        logger.info(f"Processing task with voice: {task}")

        # In a real scenario, this would:
        # 1. Connect to LiveKit room
        # 2. Listen for voice input via Whisper
        # 3. Process the task
        # 4. Respond via TTS

        result = await orchestrator.process_input(task)

        logger.info(f"Voice demo completed: {result['status']}")

        print("\n✓ Voice demo completed successfully")
        print("\nNote: For full voice interaction, ensure:")
        print("  - LiveKit server is running")
        print("  - Client is connected to the same room")
        print("  - Audio devices are properly configured")

        return True

    except Exception as e:
        logger.error(f"Voice demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run voice demo"""
    result = await demo_voice_agent()

    if result:
        print("\n🎉 Voice demo completed successfully!")
        return 0
    else:
        print("\n⚠️  Voice demo failed. Check requirements and configuration.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
