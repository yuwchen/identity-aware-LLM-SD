"""
Audio Transcription Pipeline
Uses NVIDIA NeMo models ASR transcription.
"""

from pathlib import Path
from typing import List
from pydub import AudioSegment

import nemo.collections.asr as nemo_asr

from .utils import (
    create_directory,
    delete_directory_files,
    split_audio_into_snippets
)

class AudioTranscriber:
    """Handles audio transcription using NVIDIA NeMo ASR."""
    
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
        """Initialize ASR model.
        
        Args:
            model_name: Name of the NeMo ASR model
        """
        print(f"Loading ASR model: {model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    
    def transcribe_audio(
        self,
        audio_path: str,
        snippet_duration: int,
        padding: int = 5,
        batch_size: int = 1
    ) -> List[dict]:
        """Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            snippet_duration: Duration of snippets in seconds
            padding: Padding around snippets in seconds
            batch_size: Batch size for transcription
            
        Returns:
            List of word dictionaries with timestamps
        """
        print(f"Transcribing: {audio_path}")
        
        # Load audio
        audio_segment = AudioSegment.from_file(audio_path)
        conv_id = Path(audio_path).stem
        temp_dir = f'tmp_audio/{conv_id}'
        
        # Split audio into snippets
        print("Splitting audio...")
        snippet_files = split_audio_into_snippets(
            audio_segment=audio_segment,
            output_dir=temp_dir,
            base_name=conv_id,
            snippet_duration=snippet_duration,
            padding=padding
        )
        
        # Transcribe
        print("Transcribing snippets...")
        outputs = self.model.transcribe(
            snippet_files,
            batch_size=batch_size,
            timestamps=True
        )
        
        # Combine word timestamps
        print("Processing timestamps...")
        word_timestamps = []
        for i, output in enumerate(outputs):
            start_offset = max(i * snippet_duration - padding, 0)
            
            for word in output.timestamp['word']:
                if word['start'] < snippet_duration:
                    word_timestamps.append({
                        'word': word['word'],
                        'start': word['start'] + start_offset,
                        'end': word['end'] + start_offset,
                    })
        
        # Cleanup
        delete_directory_files(temp_dir)
        
        return word_timestamps