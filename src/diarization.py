"""
Audio Diarization Pipeline
Uses NVIDIA NeMo models for speaker diarization
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import SortformerEncLabelModel

from .utils import (
    load_audio_ffmpeg,
    numpy_to_audiosegment,
    delete_directory_files,
    split_audio_into_snippets
)


class AudioDiarizer:
    """Handles speaker diarization using NVIDIA NeMo models."""
    
    def __init__(self, device: str = "auto", sample_rate: int = 16000):
        """Initialize diarization models.
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
            sample_rate: Sample rate for INPUT AUDIO.
            TODO: Think of better way to handle sampling rate when inferencing multiple audios. Stick to loading sampling rate from the audio instead of taking in user input?
        """
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        self.sample_rate_original = sample_rate
        self.sample_rate = 16000
        print(f"Loading models on {self.device}...")
        
        # Load diarization model
        self.diar_model = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_sortformer_4spk-v1"
        )
        self.diar_model.eval()
        
        # Load speaker verification model
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.speaker_model.to(self.device)
        self.speaker_model.freeze()
    
    @torch.no_grad()
    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio segment.
        
        Args:
            audio: Audio array (float32)
            
        Returns:
            Speaker embedding vector
        """
        audio_length = audio.shape[0]
        audio_tensor = torch.tensor(
            np.array([audio]), 
            device=self.device, 
            dtype=torch.float32
        )
        audio_len = torch.tensor([audio_length], device=self.device)
        
        _, emb = self.speaker_model.forward(
            input_signal=audio_tensor,
            input_signal_length=audio_len
        )
        
        return emb.detach().cpu().numpy()
    
    def compute_speaker_embeddings(
        self,
        segments: List,
        audio_array: np.ndarray,
        snippet_duration: int,
        padding: int = 5,
        embedding_dim: int = 192
    ) -> List[np.ndarray]:
        """Compute speaker embeddings for each speaker in each segment.
        
        Args:
            segments: List of diarization segments
            audio_array: Full audio array
            snippet_duration: Duration of each snippet in seconds
            padding: Padding used in snippet creation
            embedding_dim: Dimension of speaker embeddings
            
        Returns:
            List of speaker embeddings
        """
        all_embeddings = []
        
        for i, audio_seg in enumerate(segments):
            offset = max(i * snippet_duration - padding, 0)
            speaker_segments = {'speaker_0': [], 'speaker_1': [], 
                              'speaker_2': [], 'speaker_3': []}
            
            for seg in audio_seg:
                start, end, spkr = seg.split(' ')
                start, end = float(start), float(end)
                
                # Validate segment boundaries
                segment_end = snippet_duration + (2 * padding if i > 0 else padding)
                valid_until = segment_end - padding
                if end > valid_until:
                    continue
                
                # Extract audio segment
                adjusted_start = start + offset
                adjusted_end = end + offset
                segment_array = audio_array[
                    int(adjusted_start * self.sample_rate):int(adjusted_end * self.sample_rate)
                ]
                speaker_segments[spkr].append(segment_array)
            
            # Compute embeddings for each speaker
            for segments in speaker_segments.values():
                if segments:
                    combined = np.hstack(segments)
                    if combined.shape[0] > 100:
                        embedding = self.get_embedding(combined)
                    else:
                        # Audio too short. Ignore, and filter out later
                        embedding = np.zeros((1, embedding_dim))
                else:
                    embedding = np.zeros((1, embedding_dim))
                all_embeddings.append(embedding)
        
        return all_embeddings
    
    @staticmethod
    def cluster_speakers(
        embeddings: List[np.ndarray],
        n_speakers: int = 4
    ) -> dict:
        """Cluster speaker embeddings to identify unique speakers.
        
        Args:
            embeddings: List of speaker embeddings
            n_speakers: Expected number of speakers
            
        Returns:
            Dict mapping cluster labels to embedding indices
        """
        # Filter out zero vectors
        valid_indices = []
        valid_vectors = []
        
        for idx, vec in enumerate(embeddings):
            if not np.allclose(vec, 0):
                valid_indices.append(idx)
                valid_vectors.append(vec)
        
        if not valid_vectors:
            return {}
        
        # Normalize and cluster
        valid_vectors = normalize(np.vstack(valid_vectors))
        clustering = AgglomerativeClustering(
            n_clusters=min(n_speakers, len(valid_vectors))
        )
        labels = clustering.fit_predict(valid_vectors)
        
        # Map clusters to original indices
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(valid_indices[i])
        
        return clusters
    
    @staticmethod
    def assign_speaker_labels(clusters: dict, total_count: int) -> List[str]:
        """Convert cluster assignments to speaker labels.
        
        Args:
            clusters: Dict of cluster assignments
            total_count: Total number of embeddings
            
        Returns:
            List of speaker labels
        """
        labels = ['Unknown'] * total_count
        for cluster_id, indices in clusters.items():
            for idx in indices:
                labels[idx] = f'Speaker {cluster_id}'
        return labels
    
    def process_audio(
        self,
        audio_path: str,
        snippet_duration: int = 250,
        padding: int = 5,
        n_speakers: int = 4,
        batch_size: int = 1
    ) -> List[Tuple[float, float, str]]:
        """Process audio file for diarization.
        
        Args:
            audio_path: Path to audio file
            snippet_duration: Duration of snippets in seconds
            padding: Padding around snippets in seconds
            n_speakers: Expected number of speakers
            batch_size: 
                Batch size for processing # WARNING: The batch_size (>1) changes the results significantly. Why this happens is unknown. 
            
        Returns:
            List of (start, end, speaker) tuples
        """
        print(f"Processing: {audio_path}")
        
        # Load audio
        audio_array, audio_length = load_audio_ffmpeg(audio_path, sr=self.sample_rate_original)
        audio_segment = numpy_to_audiosegment(audio_array)
        
        # Create temporary directory and split audio
        conv_id = Path(audio_path).stem
        temp_dir = f'tmp_audio/{conv_id}'
        
        print("Splitting audio into snippets...")
        snippet_files = split_audio_into_snippets(
            audio_segment=audio_segment,
            output_dir=temp_dir,
            base_name=conv_id,
            snippet_duration=snippet_duration,
            padding=padding
        )

        # Perform diarization
        print("Performing diarization...")
        segments, _ = self.diar_model.diarize(
            audio=snippet_files,
            batch_size=batch_size,
            include_tensor_outputs=True
        )

        # Extract speaker embeddings
        print("Extracting speaker embeddings...")
        embeddings = self.compute_speaker_embeddings(
            segments, audio_array, snippet_duration, padding
        )
        
        # Cluster speakers
        print("Clustering speakers...")
        clusters = self.cluster_speakers(embeddings, n_speakers)
        speaker_labels = self.assign_speaker_labels(clusters, len(embeddings))
        
        # Combine segments
        print("Combining segments...")
        final_segments = self._combine_segments(
            segments, speaker_labels, snippet_duration, padding, audio_length, self.sample_rate
        )
        
        # Filter overlapping segments
        final_segments = self._filter_overlaps(final_segments)
        
        # Cleanup
        delete_directory_files(temp_dir)
        
        return final_segments
    
    @staticmethod
    def _combine_segments(
        segments: List,
        labels: List[str],
        duration: int,
        padding: int,
        audio_length: int,
        sampling_rate: int
    ) -> List[Tuple[float, float, str]]:
        """Combine and adjust segment timestamps."""
        combined = []
        
        for i, seg_list in enumerate(segments):
            offset = max(i * duration - padding, 0)
            speaker_map = {
                'speaker_0': labels[i * 4],
                'speaker_1': labels[i * 4 + 1],
                'speaker_2': labels[i * 4 + 2],
                'speaker_3': labels[i * 4 + 3],
            }
            
            for seg in seg_list:
                start, end, spkr = seg.split(' ')
                start, end = float(start), float(end)
                
                if end > audio_length / sampling_rate:
                    continue
                
                combined.append([
                    start + offset,
                    end + offset,
                    speaker_map[spkr]
                ])
        
        return sorted(combined, key=lambda x: x[0])
    
    @staticmethod
    def _filter_overlaps(
        segments: List[Tuple[float, float, str]],
        threshold: float = 0.5
    ) -> List[Tuple[float, float, str]]:
        """Remove overlapping segments for the same speaker."""
        segments = sorted(segments, key=lambda x: x[0])
        filtered = []
        
        for seg in segments:
            start, end, speaker = seg
            keep = True
            
            for prev in reversed(filtered):
                prev_start, prev_end, prev_speaker = prev
                
                if prev_speaker != speaker:
                    continue
                if prev_end <= start:
                    break
                
                overlap = min(end, prev_end) - max(start, prev_start)
                if overlap > threshold * min(end - start, prev_end - prev_start):
                    keep = False
                    break
            
            if keep:
                filtered.append(seg)
        
        return filtered