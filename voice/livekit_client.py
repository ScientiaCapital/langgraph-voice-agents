"""
LiveKit client integration for real-time voice communication.
Uses Cartesia for TTS/STT - NO OpenAI.

LiveKit provides WebRTC transport.
Cartesia provides voice synthesis and recognition.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, Callable, List, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

from livekit import rtc, api

from voice.cartesia_client import CartesiaClient, CartesiaConfig
from voice.audio_utils import (
    convert_pcm_to_wav,
    convert_s16_to_f32,
    calculate_rms_amplitude,
    chunk_audio,
)

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
    # LiveKit connection
    url: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    room_name: str = "voice-agent-room"
    participant_name: str = "agent"

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16

    # Voice activity detection
    voice_activity_threshold: float = 0.01
    silence_timeout: float = 1.5

    # Cartesia settings
    cartesia_api_key: Optional[str] = None
    cartesia_voice_id: Optional[str] = None

    def __post_init__(self):
        # Load from environment if not provided
        if self.url is None:
            self.url = os.getenv("LIVEKIT_URL")
        if self.api_key is None:
            self.api_key = os.getenv("LIVEKIT_API_KEY")
        if self.api_secret is None:
            self.api_secret = os.getenv("LIVEKIT_API_SECRET")
        if self.cartesia_api_key is None:
            self.cartesia_api_key = os.getenv("CARTESIA_API_KEY")


class AudioBuffer:
    """
    Manages audio buffering with voice activity detection.
    Collects audio chunks during speech and returns complete utterances.
    """

    def __init__(
        self,
        voice_threshold: float = 0.01,
        silence_timeout: float = 1.5,
        min_speech_duration: float = 0.3
    ):
        self.voice_threshold = voice_threshold
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration

        self._buffer: List[bytes] = []
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._last_voice_time: Optional[float] = None

    def add_audio(self, audio_data: bytes, timestamp: float) -> Optional[bytes]:
        """
        Add audio data and return complete utterance if speech ended.

        Args:
            audio_data: PCM s16le audio bytes
            timestamp: Current timestamp in seconds

        Returns:
            Complete audio utterance if speech ended, None otherwise
        """
        rms = calculate_rms_amplitude(audio_data)
        has_voice = rms > self.voice_threshold

        if has_voice:
            if not self._is_speaking:
                # Speech started
                self._is_speaking = True
                self._speech_start_time = timestamp
                self._buffer = []
                logger.debug("Speech started")

            self._buffer.append(audio_data)
            self._last_voice_time = timestamp

        elif self._is_speaking:
            # Add to buffer even during short pauses
            self._buffer.append(audio_data)

            # Check for end of speech
            if self._last_voice_time and (timestamp - self._last_voice_time) > self.silence_timeout:
                # Speech ended - check minimum duration
                speech_duration = timestamp - self._speech_start_time
                if speech_duration >= self.min_speech_duration:
                    audio = b''.join(self._buffer)
                    self._reset()
                    logger.debug(f"Speech ended, duration: {speech_duration:.2f}s")
                    return audio
                else:
                    # Too short, discard
                    logger.debug(f"Speech too short ({speech_duration:.2f}s), discarding")
                    self._reset()

        return None

    def _reset(self):
        """Reset buffer state"""
        self._buffer = []
        self._is_speaking = False
        self._speech_start_time = None
        self._last_voice_time = None

    def force_end(self) -> Optional[bytes]:
        """Force end current utterance and return audio"""
        if self._buffer:
            audio = b''.join(self._buffer)
            self._reset()
            return audio
        return None


class LiveKitCartesiaClient:
    """
    LiveKit client with Cartesia voice integration.

    Combines:
    - LiveKit: WebRTC transport for real-time audio
    - Cartesia: TTS (sonic-2) and STT (ink-whisper)

    NO OpenAI dependencies.
    """

    def __init__(self, config: Optional[LiveKitConfig] = None):
        self.config = config or LiveKitConfig()
        self.room: Optional[rtc.Room] = None
        self.status = ConnectionStatus.DISCONNECTED

        # Cartesia client for TTS/STT
        cartesia_config = CartesiaConfig(
            api_key=self.config.cartesia_api_key,
            voice_id=self.config.cartesia_voice_id or "a0e99841-438c-4a64-b679-ae501e7d6091",
            stt_sample_rate=self.config.sample_rate,
        )
        self._cartesia = CartesiaClient(cartesia_config)

        # Audio management
        self._audio_buffer = AudioBuffer(
            voice_threshold=self.config.voice_activity_threshold,
            silence_timeout=self.config.silence_timeout,
        )

        # Audio publishing
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None

        # Event callbacks
        self.on_transcription: Optional[Callable[[str], Any]] = None
        self.on_participant_connected: Optional[Callable[[rtc.Participant], Any]] = None
        self.on_participant_disconnected: Optional[Callable[[rtc.Participant], Any]] = None

        # Processing state
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self) -> bool:
        """Connect to LiveKit room and initialize Cartesia"""
        if not self.config.url or not self.config.api_key or not self.config.api_secret:
            raise ValueError(
                "LiveKit configuration incomplete. Set LIVEKIT_URL, "
                "LIVEKIT_API_KEY, and LIVEKIT_API_SECRET."
            )

        try:
            self.status = ConnectionStatus.CONNECTING

            # Connect Cartesia
            await self._cartesia.connect()

            # Create and connect LiveKit room
            self.room = rtc.Room()
            self._setup_event_handlers()

            token = self._generate_token()
            await self.room.connect(self.config.url, token)

            # Set up audio publishing
            await self._setup_audio_track()

            self.status = ConnectionStatus.CONNECTED
            self._running = True

            logger.info(f"Connected to LiveKit room: {self.config.room_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.status = ConnectionStatus.FAILED
            return False

    async def disconnect(self):
        """Disconnect from LiveKit and Cartesia"""
        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self.room:
            await self.room.disconnect()
            self.room = None

        await self._cartesia.disconnect()

        self.status = ConnectionStatus.DISCONNECTED
        logger.info("Disconnected from LiveKit")

    def _generate_token(self) -> str:
        """Generate JWT token for LiveKit authentication"""
        token = api.AccessToken(self.config.api_key, self.config.api_secret)
        token.with_identity(self.config.participant_name)
        token.with_name(self.config.participant_name)

        grants = api.VideoGrants(
            room_join=True,
            room=self.config.room_name,
            can_publish=True,
            can_subscribe=True
        )
        token.with_grants(grants)

        return token.to_jwt()

    def _setup_event_handlers(self):
        """Set up LiveKit room event handlers"""

        @self.room.on("participant_connected")
        def on_connected(participant: rtc.Participant):
            logger.info(f"Participant connected: {participant.identity}")
            if self.on_participant_connected:
                asyncio.create_task(
                    self._safe_callback(self.on_participant_connected, participant)
                )

        @self.room.on("participant_disconnected")
        def on_disconnected(participant: rtc.Participant):
            logger.info(f"Participant disconnected: {participant.identity}")
            if self.on_participant_disconnected:
                asyncio.create_task(
                    self._safe_callback(self.on_participant_disconnected, participant)
                )

        @self.room.on("track_subscribed")
        def on_track(track, publication, participant):
            if isinstance(track, rtc.AudioTrack):
                logger.info(f"Subscribed to audio from {participant.identity}")
                asyncio.create_task(self._handle_audio_track(track))

    async def _safe_callback(self, callback: Callable, *args):
        """Execute callback safely with error handling"""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def _setup_audio_track(self):
        """Set up local audio track for publishing"""
        self._audio_source = rtc.AudioSource(
            sample_rate=self.config.sample_rate,
            num_channels=self.config.channels
        )

        self._audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-voice",
            self._audio_source
        )

        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE
        )

        await self.room.local_participant.publish_track(
            self._audio_track,
            options
        )

        logger.info("Audio track published")

    async def _handle_audio_track(self, track: rtc.AudioTrack):
        """Handle incoming audio track and process with Cartesia STT"""
        stream = rtc.AudioStream(track)
        current_time = 0.0
        frame_duration = 0.02  # 20ms frames

        async for frame in stream:
            if not self._running:
                break

            audio_data = frame.data.tobytes()
            current_time += frame_duration

            # Check for complete utterance
            utterance = self._audio_buffer.add_audio(audio_data, current_time)

            if utterance:
                # Transcribe with Cartesia
                await self._process_utterance(utterance)

    async def _process_utterance(self, audio_data: bytes):
        """Process complete utterance through Cartesia STT"""
        try:
            transcription = await self._cartesia.listen(audio_data)

            if transcription and self.on_transcription:
                await self._safe_callback(self.on_transcription, transcription)

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    async def speak(self, text: str) -> bool:
        """
        Synthesize text and publish to LiveKit room.

        Args:
            text: Text to speak

        Returns:
            True if successful
        """
        if not self._audio_source:
            logger.error("Audio source not available")
            return False

        try:
            # Stream audio chunks from Cartesia
            async for chunk in self._cartesia.speak(text):
                if not self._running:
                    break

                # Cartesia returns s16le, create audio frame
                # LiveKit expects specific frame format
                samples_per_channel = len(chunk) // 2  # 2 bytes per s16 sample

                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=22050,  # Cartesia TTS sample rate
                    num_channels=1,
                    samples_per_channel=samples_per_channel
                )

                await self._audio_source.capture_frame(frame)

            logger.debug(f"Spoke: {text[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return False

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data using Cartesia STT.

        Args:
            audio_data: PCM s16le audio bytes

        Returns:
            Transcribed text
        """
        return await self._cartesia.listen(audio_data)

    def is_connected(self) -> bool:
        """Check if connected to LiveKit"""
        return self.status == ConnectionStatus.CONNECTED

    def get_participants(self) -> List[rtc.Participant]:
        """Get list of room participants"""
        if not self.room:
            return []
        return list(self.room.remote_participants.values())

    def get_room_info(self) -> Dict[str, Any]:
        """Get current room information"""
        if not self.room:
            return {"status": self.status.value}

        return {
            "room_name": self.config.room_name,
            "status": self.status.value,
            "participant_count": len(self.room.remote_participants) + 1,
            "local_participant": self.config.participant_name,
            "remote_participants": [
                p.identity for p in self.room.remote_participants.values()
            ]
        }

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# Factory function for easy creation
async def create_voice_session(
    room_name: str,
    participant_name: str = "agent",
    on_transcription: Optional[Callable[[str], Any]] = None,
    livekit_url: Optional[str] = None,
    livekit_api_key: Optional[str] = None,
    livekit_api_secret: Optional[str] = None,
    cartesia_api_key: Optional[str] = None,
    cartesia_voice_id: Optional[str] = None,
) -> LiveKitCartesiaClient:
    """
    Create and connect a voice session.

    Args:
        room_name: LiveKit room name
        participant_name: Agent's participant name
        on_transcription: Callback for transcribed speech
        livekit_url: LiveKit server URL
        livekit_api_key: LiveKit API key
        livekit_api_secret: LiveKit API secret
        cartesia_api_key: Cartesia API key
        cartesia_voice_id: Cartesia voice ID for TTS

    Returns:
        Connected LiveKitCartesiaClient
    """
    config = LiveKitConfig(
        url=livekit_url,
        api_key=livekit_api_key,
        api_secret=livekit_api_secret,
        room_name=room_name,
        participant_name=participant_name,
        cartesia_api_key=cartesia_api_key,
        cartesia_voice_id=cartesia_voice_id,
    )

    client = LiveKitCartesiaClient(config)
    client.on_transcription = on_transcription

    if await client.connect():
        return client
    else:
        raise RuntimeError("Failed to create voice session")
