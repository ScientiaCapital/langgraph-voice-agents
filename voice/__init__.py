"""
Voice integration using LiveKit for real-time audio communication.

This package provides LiveKit WebRTC client integration with:
- OpenAI Whisper for speech-to-text
- OpenAI TTS for text-to-speech
- Voice Activity Detection
- Real-time audio processing
"""

from .livekit_client import LiveKitClient, LiveKitConfig, AudioProcessor

__all__ = [
    "LiveKitClient",
    "LiveKitConfig",
    "AudioProcessor",
]
