"""
Voice module for LangGraph agents.
Provides Cartesia TTS/STT integration with LiveKit WebRTC transport.

NO OpenAI - Cartesia is the exclusive voice provider.
"""

from voice.cartesia_client import (
    CartesiaConfig,
    CartesiaTTS,
    CartesiaSTT,
    CartesiaClient,
    ConnectionStatus as CartesiaConnectionStatus,
    WordTimestamp,
    TranscriptResult,
)
from voice.livekit_client import (
    LiveKitConfig,
    LiveKitCartesiaClient,
    AudioBuffer,
    ConnectionStatus as LiveKitConnectionStatus,
    create_voice_session,
)
from voice.audio_utils import (
    AudioFormat,
    AudioConfig,
    convert_pcm_to_wav,
    convert_wav_to_pcm,
    convert_f32_to_s16,
    convert_s16_to_f32,
    chunk_audio,
    calculate_duration_ms,
    calculate_rms_amplitude,
)

__all__ = [
    # Cartesia Client
    "CartesiaConfig",
    "CartesiaTTS",
    "CartesiaSTT",
    "CartesiaClient",
    "CartesiaConnectionStatus",
    "WordTimestamp",
    "TranscriptResult",
    # LiveKit + Cartesia Integration
    "LiveKitConfig",
    "LiveKitCartesiaClient",
    "AudioBuffer",
    "LiveKitConnectionStatus",
    "create_voice_session",
    # Audio Utilities
    "AudioFormat",
    "AudioConfig",
    "convert_pcm_to_wav",
    "convert_wav_to_pcm",
    "convert_f32_to_s16",
    "convert_s16_to_f32",
    "chunk_audio",
    "calculate_duration_ms",
    "calculate_rms_amplitude",
]
