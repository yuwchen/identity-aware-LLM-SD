"""
Utility functions for audio processing and file operations.
"""

import os
import io
import subprocess
import numpy as np
import librosa
from pydub import AudioSegment


def load_audio_ffmpeg(file: str, resample: bool = True, sr: int = 16000, target_sr: int = 16000) -> tuple:
    """Load audio file using ffmpeg and optionally resample.
    
    Args:
        file: Path to audio file
        resample: Whether to resample audio
        sr: Source sample rate
        
    Returns:
        Tuple of (audio_array, audio_length)
        
    Raises:
        RuntimeError: If ffmpeg fails to load audio
    """
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
    if resample and sr != target_sr:
        audio = librosa.core.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio, audio.shape[0]


def numpy_to_audiosegment(
    audio: np.ndarray,
    sample_rate: int = 16000,
    channels: int = 1
) -> AudioSegment:
    """Convert numpy array to pydub AudioSegment.
    
    Args:
        audio: Audio array (float32, range [-1, 1])
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        
    Returns:
        AudioSegment object
    """
    # Convert float32 [-1, 1] to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create byte buffer
    byte_buffer = io.BytesIO()
    byte_buffer.write(audio_int16.tobytes())
    byte_buffer.seek(0)
    
    # Load with pydub
    audio_segment = AudioSegment.from_raw(
        byte_buffer,
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels
    )
    
    return audio_segment


def create_directory(dir_path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        dir_path: Path to directory
    """
    os.makedirs(dir_path, exist_ok=True)


def delete_directory_files(dir_path: str) -> None:
    """Delete all files in a directory (keeps subdirectories).
    
    Args:
        dir_path: Path to directory
    """
    if not os.path.isdir(dir_path):
        print(f"Warning: Directory not found at {dir_path}")
        return
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")


def split_audio_into_snippets(
    audio_segment: AudioSegment,
    output_dir: str,
    base_name: str,
    snippet_duration: int,
    padding: int = 5
) -> list:
    """Split audio into overlapping snippets and save to disk.
    
    Args:
        audio_segment: AudioSegment object to split
        output_dir: Directory to save snippets
        base_name: Base name for snippet files
        snippet_duration: Duration of each snippet in seconds
        padding: Padding to add before/after each snippet in seconds
        
    Returns:
        List of snippet file paths, sorted by index
    """
    create_directory(output_dir)
    delete_directory_files(output_dir)
    
    i = 0
    while True:
        start_ms = max((i * snippet_duration - padding) * 1000, 0)
        end_ms = ((i + 1) * snippet_duration + padding) * 1000
        snippet = audio_segment[start_ms:end_ms]
        
        if snippet.duration_seconds <= 0:
            break
        
        snippet_path = f"{output_dir}/{base_name}_{i}.wav"
        snippet.export(snippet_path, format="wav")
        i += 1
    
    # Get and sort snippet files
    snippet_files = [
        f"{output_dir}/{f}" for f in os.listdir(output_dir)
        if f.endswith('.wav')
    ]
    snippet_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    return snippet_files