"""
LiveKit client integration for real-time voice communication in agents.
Handles WebRTC connections, audio processing, and room management.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass
from enum import Enum
import json
import wave
import io

from livekit import rtc, api
import numpy as np
from openai import AsyncOpenAI
import asyncio
import websockets

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """LiveKit connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class LiveKitConfig:
    """Configuration for LiveKit client"""
    url: str
    api_key: str
    api_secret: str
    room_name: str
    participant_name: str

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16

    # Voice processing
    voice_activity_threshold: float = 0.01
    silence_timeout: float = 2.0

    # TTS/STT settings
    openai_api_key: Optional[str] = None
    tts_voice: str = "alloy"
    tts_model: str = "tts-1"
    stt_model: str = "whisper-1"


class AudioProcessor:
    """Handles audio processing for voice interactions"""

    def __init__(self, config: LiveKitConfig):
        self.config = config
        self.audio_buffer = []
        self.is_recording = False
        self.last_audio_time = 0

    def detect_voice_activity(self, audio_data: bytes) -> bool:
        """Simple voice activity detection"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate RMS (root mean square) for amplitude
            rms = np.sqrt(np.mean(audio_array**2))

            # Normalize to 0-1 range
            normalized_rms = rms / 32768.0

            return normalized_rms > self.config.voice_activity_threshold

        except Exception as e:
            logger.error(f"Voice activity detection failed: {e}")
            return False

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.audio_buffer = []
        self.last_audio_time = asyncio.get_event_loop().time()

    def stop_recording(self) -> bytes:
        """Stop recording and return audio data"""
        self.is_recording = False

        if not self.audio_buffer:
            return b""

        # Combine all audio chunks
        combined_audio = b"".join(self.audio_buffer)
        self.audio_buffer = []

        return combined_audio

    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk to buffer"""
        if self.is_recording:
            self.audio_buffer.append(audio_data)
            self.last_audio_time = asyncio.get_event_loop().time()

    def check_silence_timeout(self) -> bool:
        """Check if silence timeout has been reached"""
        if not self.is_recording:
            return False

        current_time = asyncio.get_event_loop().time()
        return (current_time - self.last_audio_time) > self.config.silence_timeout

    def create_wav_file(self, audio_data: bytes) -> bytes:
        """Create WAV file from raw audio data"""
        try:
            buffer = io.BytesIO()

            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(self.config.bits_per_sample // 8)
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_data)

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"WAV file creation failed: {e}")
            return b""


class VoiceProcessor:
    """Handles speech-to-text and text-to-speech operations"""

    def __init__(self, config: LiveKitConfig):
        self.config = config
        self.openai_client = None

        if config.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)

    async def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text using OpenAI Whisper"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not configured")

        try:
            # Create a file-like object from audio data
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"

            # Use OpenAI Whisper for transcription
            transcript = await self.openai_client.audio.transcriptions.create(
                model=self.config.stt_model,
                file=audio_file,
                response_format="text"
            )

            logger.debug(f"Transcribed: {transcript}")
            return transcript.strip()

        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            return ""

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using OpenAI TTS"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not configured")

        try:
            response = await self.openai_client.audio.speech.create(
                model=self.config.tts_model,
                voice=self.config.tts_voice,
                input=text,
                response_format="wav"
            )

            return response.content

        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return b""


class LiveKitClient:
    """Main LiveKit client for agent voice interactions"""

    def __init__(self, config: LiveKitConfig):
        self.config = config
        self.room = None
        self.status = ConnectionStatus.DISCONNECTED

        # Audio processing
        self.audio_processor = AudioProcessor(config)
        self.voice_processor = VoiceProcessor(config)

        # Event handlers
        self.on_audio_received: Optional[Callable] = None
        self.on_transcription_complete: Optional[Callable] = None
        self.on_participant_connected: Optional[Callable] = None
        self.on_participant_disconnected: Optional[Callable] = None

        # Internal state
        self._audio_track = None
        self._audio_source = None
        self._current_session = None

    async def connect(self) -> bool:
        """Connect to LiveKit room"""
        try:
            self.status = ConnectionStatus.CONNECTING

            # Create room instance
            self.room = rtc.Room()

            # Set up event handlers
            self._setup_event_handlers()

            # Connect to room
            await self.room.connect(
                url=self.config.url,
                token=self._generate_token()
            )

            # Set up audio track
            await self._setup_audio()

            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to LiveKit room: {self.config.room_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to LiveKit: {e}")
            self.status = ConnectionStatus.FAILED
            return False

    async def disconnect(self):
        """Disconnect from LiveKit room"""
        try:
            if self.room:
                await self.room.disconnect()
                self.room = None

            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from LiveKit")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def _generate_token(self) -> str:
        """Generate JWT token for LiveKit authentication"""
        from livekit.api import AccessToken, VideoGrants

        token = AccessToken(self.config.api_key, self.config.api_secret)
        token.with_identity(self.config.participant_name)
        token.with_name(self.config.participant_name)

        grants = VideoGrants(
            room_join=True,
            room=self.config.room_name,
            can_publish=True,
            can_subscribe=True
        )
        token.with_grants(grants)

        return token.to_jwt()

    def _setup_event_handlers(self):
        """Set up LiveKit event handlers"""

        @self.room.on("participant_connected")
        def on_participant_connected(participant):
            logger.info(f"Participant connected: {participant.identity}")
            if self.on_participant_connected:
                asyncio.create_task(self.on_participant_connected(participant))

        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            logger.info(f"Participant disconnected: {participant.identity}")
            if self.on_participant_disconnected:
                asyncio.create_task(self.on_participant_disconnected(participant))

        @self.room.on("track_published")
        def on_track_published(publication, participant):
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Audio track published by {participant.identity}")

        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if isinstance(track, rtc.AudioTrack):
                logger.info(f"Subscribed to audio track from {participant.identity}")
                asyncio.create_task(self._handle_audio_track(track))

    async def _setup_audio(self):
        """Set up audio publishing"""
        try:
            # Create audio source
            self._audio_source = rtc.AudioSource(
                sample_rate=self.config.sample_rate,
                num_channels=self.config.channels
            )

            # Create audio track
            self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                "agent-audio", self._audio_source
            )

            # Publish audio track
            await self.room.local_participant.publish_track(
                self._audio_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )

            logger.info("Audio track published")

        except Exception as e:
            logger.error(f"Failed to set up audio: {e}")

    async def _handle_audio_track(self, track: rtc.AudioTrack):
        """Handle incoming audio track"""
        try:
            audio_stream = rtc.AudioStream(track)

            async for frame in audio_stream:
                audio_data = frame.data.tobytes()

                # Process audio data
                if self.audio_processor.detect_voice_activity(audio_data):
                    if not self.audio_processor.is_recording:
                        self.audio_processor.start_recording()

                    self.audio_processor.add_audio_chunk(audio_data)

                elif self.audio_processor.is_recording:
                    # Check for silence timeout
                    if self.audio_processor.check_silence_timeout():
                        audio_recording = self.audio_processor.stop_recording()

                        if audio_recording:
                            await self._process_voice_input(audio_recording)

                # Call custom handler if set
                if self.on_audio_received:
                    await self.on_audio_received(audio_data)

        except Exception as e:
            logger.error(f"Error handling audio track: {e}")

    async def _process_voice_input(self, audio_data: bytes):
        """Process recorded voice input"""
        try:
            # Convert to WAV format
            wav_data = self.audio_processor.create_wav_file(audio_data)

            if not wav_data:
                return

            # Transcribe audio
            transcription = await self.voice_processor.speech_to_text(wav_data)

            if transcription and self.on_transcription_complete:
                await self.on_transcription_complete(transcription)

        except Exception as e:
            logger.error(f"Error processing voice input: {e}")

    async def speak(self, text: str) -> bool:
        """Convert text to speech and play it"""
        try:
            if not self._audio_source:
                logger.error("Audio source not available")
                return False

            # Generate speech
            audio_data = await self.voice_processor.text_to_speech(text)

            if not audio_data:
                return False

            # Convert to audio frame format
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Create audio frame
            frame = rtc.AudioFrame(
                data=audio_array,
                sample_rate=self.config.sample_rate,
                num_channels=self.config.channels,
                samples_per_channel=len(audio_array) // self.config.channels
            )

            # Publish audio frame
            await self._audio_source.capture_frame(frame)

            logger.debug(f"Spoke: {text}")
            return True

        except Exception as e:
            logger.error(f"Failed to speak: {e}")
            return False

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio data to text"""
        wav_data = self.audio_processor.create_wav_file(audio_data)
        return await self.voice_processor.speech_to_text(wav_data)

    async def synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech from text"""
        return await self.voice_processor.text_to_speech(text)

    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self.status == ConnectionStatus.CONNECTED

    def get_participants(self) -> List[rtc.Participant]:
        """Get list of participants in room"""
        if not self.room:
            return []

        return list(self.room.participants.values())

    def get_room_info(self) -> Dict[str, Any]:
        """Get room information"""
        if not self.room:
            return {}

        return {
            "room_name": self.config.room_name,
            "participant_count": len(self.room.participants),
            "connection_status": self.status.value,
            "local_participant": self.room.local_participant.identity if self.room.local_participant else None
        }


# Utility functions

def create_room_name(agent_type: str, session_id: str) -> str:
    """Create standardized room name"""
    return f"agent-framework-{agent_type}-{session_id}"


async def create_livekit_client(
    agent_type: str,
    session_id: str,
    participant_name: str,
    livekit_url: str,
    api_key: str,
    api_secret: str,
    openai_api_key: Optional[str] = None
) -> LiveKitClient:
    """Factory function to create and connect LiveKit client"""

    config = LiveKitConfig(
        url=livekit_url,
        api_key=api_key,
        api_secret=api_secret,
        room_name=create_room_name(agent_type, session_id),
        participant_name=participant_name,
        openai_api_key=openai_api_key
    )

    client = LiveKitClient(config)

    if await client.connect():
        return client
    else:
        raise RuntimeError("Failed to connect to LiveKit")


# Event handler decorators

def on_voice_command(client: LiveKitClient):
    """Decorator for voice command handlers"""
    def decorator(func):
        async def wrapper(transcription: str):
            return await func(transcription)

        client.on_transcription_complete = wrapper
        return func

    return decorator


def on_participant_join(client: LiveKitClient):
    """Decorator for participant join handlers"""
    def decorator(func):
        async def wrapper(participant):
            return await func(participant)

        client.on_participant_connected = wrapper
        return func

    return decorator