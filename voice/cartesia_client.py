"""
Cartesia voice client for TTS (Text-to-Speech) and STT (Speech-to-Text).
Uses WebSocket streaming for low-latency real-time voice processing.

NO OpenAI - Cartesia is the exclusive voice provider.
"""

import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, List
from enum import Enum

from cartesia import Cartesia, AsyncCartesia

from voice.audio_utils import convert_f32_to_s16, AudioFormat

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """WebSocket connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class WordTimestamp:
    """Word-level timestamp from TTS or STT"""
    word: str
    start: float  # Start time in seconds
    end: float    # End time in seconds


@dataclass
class TranscriptResult:
    """Result from STT transcription"""
    text: str
    is_final: bool
    words: List[WordTimestamp] = field(default_factory=list)


@dataclass
class CartesiaConfig:
    """Configuration for Cartesia voice client"""
    api_key: Optional[str] = None  # Defaults to CARTESIA_API_KEY env var

    # TTS Settings
    tts_model: str = "sonic-2"
    voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Default voice
    tts_sample_rate: int = 22050

    # STT Settings
    stt_model: str = "ink-whisper"
    stt_sample_rate: int = 16000
    language: str = "en"

    # Voice Activity Detection
    min_volume: float = 0.1
    max_silence_duration_secs: float = 0.4

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("CARTESIA_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Cartesia API key required. Set CARTESIA_API_KEY environment variable."
                )


class CartesiaTTS:
    """
    Cartesia Text-to-Speech client with WebSocket streaming.

    Features:
    - Real-time audio streaming as text is processed
    - Word-level timestamps for lip sync and captions
    - Configurable voice and sample rate
    """

    def __init__(self, config: CartesiaConfig):
        self.config = config
        self._client: Optional[AsyncCartesia] = None
        self._ws = None
        self.status = ConnectionStatus.DISCONNECTED

    async def connect(self):
        """Establish WebSocket connection for TTS"""
        if self.status == ConnectionStatus.CONNECTED:
            return

        self.status = ConnectionStatus.CONNECTING
        try:
            self._client = AsyncCartesia(api_key=self.config.api_key)
            self._ws = await self._client.tts.websocket()
            self.status = ConnectionStatus.CONNECTED
            logger.info("Cartesia TTS WebSocket connected")
        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect Cartesia TTS: {e}")
            raise

    async def disconnect(self):
        """Close WebSocket connection"""
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._client:
            await self._client.close()
            self._client = None
        self.status = ConnectionStatus.DISCONNECTED
        logger.info("Cartesia TTS WebSocket disconnected")

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from text with streaming audio output.

        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID override

        Yields:
            Audio chunks as bytes (PCM s16le format)
        """
        if self.status != ConnectionStatus.CONNECTED:
            await self.connect()

        try:
            output = await self._ws.send(
                model_id=self.config.tts_model,
                transcript=text,
                voice={"id": voice_id or self.config.voice_id},
                stream=True,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": self.config.tts_sample_rate
                }
            )

            async for chunk in output:
                if chunk.audio is not None:
                    # Convert f32 to s16 for compatibility
                    audio_s16 = convert_f32_to_s16(chunk.audio)
                    yield audio_s16

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise

    async def synthesize_with_timestamps(
        self,
        text: str,
        voice_id: Optional[str] = None
    ) -> tuple[bytes, List[WordTimestamp]]:
        """
        Synthesize speech with word-level timestamps.

        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID override

        Returns:
            Tuple of (complete audio bytes, list of word timestamps)
        """
        if self.status != ConnectionStatus.CONNECTED:
            await self.connect()

        audio_chunks = []
        all_words = []
        all_starts = []
        all_ends = []

        try:
            output = await self._ws.send(
                model_id=self.config.tts_model,
                transcript=text,
                voice={"id": voice_id or self.config.voice_id},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": self.config.tts_sample_rate
                },
                add_timestamps=True,
                stream=True
            )

            async for chunk in output:
                if chunk.audio is not None:
                    audio_s16 = convert_f32_to_s16(chunk.audio)
                    audio_chunks.append(audio_s16)

                if chunk.word_timestamps is not None:
                    all_words.extend(chunk.word_timestamps.words)
                    all_starts.extend(chunk.word_timestamps.start)
                    all_ends.extend(chunk.word_timestamps.end)

        except Exception as e:
            logger.error(f"TTS synthesis with timestamps error: {e}")
            raise

        # Build word timestamp objects
        timestamps = [
            WordTimestamp(word=w, start=s, end=e)
            for w, s, e in zip(all_words, all_starts, all_ends)
        ]

        return b''.join(audio_chunks), timestamps


class CartesiaSTT:
    """
    Cartesia Speech-to-Text client with WebSocket streaming.

    Features:
    - Real-time transcription with partial results
    - Voice Activity Detection (VAD)
    - Word-level timestamps
    - Automatic endpointing
    """

    def __init__(self, config: CartesiaConfig):
        self.config = config
        self._client: Optional[Cartesia] = None
        self._ws = None
        self.status = ConnectionStatus.DISCONNECTED

    async def connect(self):
        """Establish WebSocket connection for STT"""
        if self.status == ConnectionStatus.CONNECTED:
            return

        self.status = ConnectionStatus.CONNECTING
        try:
            # Note: STT uses sync client with websocket
            self._client = Cartesia(api_key=self.config.api_key)
            self._ws = self._client.stt.websocket(
                model=self.config.stt_model,
                language=self.config.language,
                encoding="pcm_s16le",
                sample_rate=self.config.stt_sample_rate,
                min_volume=self.config.min_volume,
                max_silence_duration_secs=self.config.max_silence_duration_secs
            )
            self.status = ConnectionStatus.CONNECTED
            logger.info("Cartesia STT WebSocket connected")
        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect Cartesia STT: {e}")
            raise

    async def disconnect(self):
        """Close WebSocket connection"""
        if self._ws:
            self._ws.close()
            self._ws = None
        if self._client:
            self._client.close()
            self._client = None
        self.status = ConnectionStatus.DISCONNECTED
        logger.info("Cartesia STT WebSocket disconnected")

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe complete audio data.

        Args:
            audio_data: PCM s16le audio bytes

        Returns:
            Complete transcript text
        """
        if self.status != ConnectionStatus.CONNECTED:
            await self.connect()

        # Split audio into chunks (20ms chunks for 16kHz, 16-bit audio)
        chunk_size = 640  # 16000 * 0.02 * 2 bytes
        audio_chunks = [
            audio_data[i:i+chunk_size]
            for i in range(0, len(audio_data), chunk_size)
        ]

        # Send audio chunks
        for chunk in audio_chunks:
            self._ws.send(chunk)

        # Signal completion
        self._ws.send("finalize")
        self._ws.send("done")

        # Collect transcript
        full_transcript = ""
        for result in self._ws.receive():
            if result['type'] == 'transcript':
                if result['is_final']:
                    full_transcript += result['text'] + " "
            elif result['type'] == 'done':
                break

        return full_transcript.strip()

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[TranscriptResult, None]:
        """
        Transcribe streaming audio with real-time results.

        Args:
            audio_stream: Async generator yielding audio chunks

        Yields:
            TranscriptResult objects with partial and final results
        """
        if self.status != ConnectionStatus.CONNECTED:
            await self.connect()

        # Start background task to send audio
        async def send_audio():
            try:
                async for chunk in audio_stream:
                    self._ws.send(chunk)
                    await asyncio.sleep(0.02)  # Small delay for real-time simulation
                self._ws.send("finalize")
                self._ws.send("done")
            except Exception as e:
                logger.error(f"Error sending audio to STT: {e}")

        # Start sending in background
        send_task = asyncio.create_task(send_audio())

        # Receive transcripts
        try:
            for result in self._ws.receive():
                if result['type'] == 'transcript':
                    words = []
                    if 'words' in result and result['words']:
                        words = [
                            WordTimestamp(
                                word=w['word'],
                                start=w['start'],
                                end=w['end']
                            )
                            for w in result['words']
                        ]

                    yield TranscriptResult(
                        text=result['text'],
                        is_final=result['is_final'],
                        words=words
                    )
                elif result['type'] == 'done':
                    break

        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass


class CartesiaClient:
    """
    Unified Cartesia client for voice processing.
    Combines TTS and STT functionality with shared configuration.
    """

    def __init__(self, config: Optional[CartesiaConfig] = None):
        self.config = config or CartesiaConfig()
        self.tts = CartesiaTTS(self.config)
        self.stt = CartesiaSTT(self.config)

    async def connect(self):
        """Connect both TTS and STT"""
        await asyncio.gather(
            self.tts.connect(),
            self.stt.connect()
        )
        logger.info("Cartesia client fully connected")

    async def disconnect(self):
        """Disconnect both TTS and STT"""
        await asyncio.gather(
            self.tts.disconnect(),
            self.stt.disconnect()
        )
        logger.info("Cartesia client fully disconnected")

    async def speak(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Convert text to speech audio stream.

        Args:
            text: Text to speak

        Yields:
            Audio chunks (PCM s16le)
        """
        async for chunk in self.tts.synthesize(text):
            yield chunk

    async def listen(self, audio_data: bytes) -> str:
        """
        Convert speech audio to text.

        Args:
            audio_data: Audio to transcribe (PCM s16le)

        Returns:
            Transcribed text
        """
        return await self.stt.transcribe_audio(audio_data)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
