"""
Audio utilities for voice processing.
Handles format conversion between PCM and WAV.
"""

import io
import struct
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""
    PCM_S16LE = "pcm_s16le"  # 16-bit signed little-endian PCM
    PCM_F32LE = "pcm_f32le"  # 32-bit float little-endian PCM
    WAV = "wav"


@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 22050
    channels: int = 1
    bits_per_sample: int = 16
    format: AudioFormat = AudioFormat.PCM_S16LE


def convert_pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 22050,
    channels: int = 1,
    bits_per_sample: int = 16
) -> bytes:
    """
    Convert raw PCM audio data to WAV format.

    Args:
        pcm_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz (default 22050 for Cartesia)
        channels: Number of audio channels (1 for mono)
        bits_per_sample: Bits per sample (16 for PCM_S16LE)

    Returns:
        WAV formatted audio bytes
    """
    buffer = io.BytesIO()

    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    buffer.seek(0)
    return buffer.read()


def convert_wav_to_pcm(wav_data: bytes) -> tuple[bytes, int, int, int]:
    """
    Convert WAV audio data to raw PCM format.

    Args:
        wav_data: WAV formatted audio bytes

    Returns:
        Tuple of (pcm_data, sample_rate, channels, bits_per_sample)
    """
    buffer = io.BytesIO(wav_data)

    with wave.open(buffer, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        bits_per_sample = wav_file.getsampwidth() * 8
        pcm_data = wav_file.readframes(wav_file.getnframes())

    return pcm_data, sample_rate, channels, bits_per_sample


def convert_f32_to_s16(f32_data: bytes) -> bytes:
    """
    Convert 32-bit float PCM to 16-bit signed PCM.
    Cartesia TTS returns f32le, but playback often needs s16le.

    Args:
        f32_data: 32-bit float PCM data

    Returns:
        16-bit signed PCM data
    """
    # Unpack as 32-bit floats
    num_samples = len(f32_data) // 4
    floats = struct.unpack(f'<{num_samples}f', f32_data)

    # Convert to 16-bit signed integers
    # Clamp values to [-1.0, 1.0] then scale to [-32768, 32767]
    s16_samples = []
    for f in floats:
        clamped = max(-1.0, min(1.0, f))
        s16_samples.append(int(clamped * 32767))

    # Pack as 16-bit signed integers
    return struct.pack(f'<{len(s16_samples)}h', *s16_samples)


def convert_s16_to_f32(s16_data: bytes) -> bytes:
    """
    Convert 16-bit signed PCM to 32-bit float PCM.

    Args:
        s16_data: 16-bit signed PCM data

    Returns:
        32-bit float PCM data
    """
    # Unpack as 16-bit signed integers
    num_samples = len(s16_data) // 2
    s16_samples = struct.unpack(f'<{num_samples}h', s16_data)

    # Convert to 32-bit floats normalized to [-1.0, 1.0]
    floats = [s / 32768.0 for s in s16_samples]

    # Pack as 32-bit floats
    return struct.pack(f'<{len(floats)}f', *floats)


def chunk_audio(
    audio_data: bytes,
    chunk_size_ms: int = 20,
    sample_rate: int = 16000,
    bytes_per_sample: int = 2
) -> list[bytes]:
    """
    Split audio data into chunks of specified duration.
    Useful for streaming audio to STT services.

    Args:
        audio_data: Raw audio bytes
        chunk_size_ms: Chunk duration in milliseconds
        sample_rate: Audio sample rate in Hz
        bytes_per_sample: Bytes per audio sample

    Returns:
        List of audio chunks
    """
    # Calculate bytes per chunk
    samples_per_chunk = int(sample_rate * chunk_size_ms / 1000)
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    chunks = []
    for i in range(0, len(audio_data), bytes_per_chunk):
        chunk = audio_data[i:i + bytes_per_chunk]
        if chunk:
            chunks.append(chunk)

    return chunks


def calculate_duration_ms(
    audio_data: bytes,
    sample_rate: int = 16000,
    bytes_per_sample: int = 2
) -> float:
    """
    Calculate duration of audio in milliseconds.

    Args:
        audio_data: Raw audio bytes
        sample_rate: Audio sample rate in Hz
        bytes_per_sample: Bytes per audio sample

    Returns:
        Duration in milliseconds
    """
    num_samples = len(audio_data) // bytes_per_sample
    return (num_samples / sample_rate) * 1000


def calculate_rms_amplitude(audio_data: bytes, bytes_per_sample: int = 2) -> float:
    """
    Calculate RMS amplitude of audio data.
    Useful for voice activity detection.

    Args:
        audio_data: Raw audio bytes (assumed to be s16le)
        bytes_per_sample: Bytes per sample

    Returns:
        RMS amplitude normalized to [0, 1]
    """
    if not audio_data:
        return 0.0

    num_samples = len(audio_data) // bytes_per_sample
    samples = struct.unpack(f'<{num_samples}h', audio_data)

    # Calculate RMS
    sum_squares = sum(s * s for s in samples)
    rms = (sum_squares / num_samples) ** 0.5

    # Normalize to [0, 1]
    return rms / 32768.0
