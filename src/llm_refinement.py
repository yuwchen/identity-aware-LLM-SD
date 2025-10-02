"""
Post-processing and Refinement Pipeline
Uses LLM and speaker embeddings to refine diarization results
"""

import os
import re
import json
import numpy as np
import torch
import faiss
import soundfile as sf
import textdistance
from typing import List, Dict, Tuple, Optional
from collections import Counter
from tqdm import tqdm

import nemo.collections.asr as nemo_asr
from openai import OpenAI

from .utils import load_audio_ffmpeg


class DiarizationRefiner:
    """Handles post-processing and refinement of diarization results."""
    
    def __init__(
        self, 
        openai_api_key: str,
        device: str = "auto",
        speaker_model = None,
        asr_model = None,
        speaker_model_name: str = "nvidia/speakerverification_en_titanet_large",
        asr_model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        openai_model: str = "gpt-4.1"
    ):
        """Initialize refinement models.
        
        Args:
            openai_api_key: OpenAI API key for GPT processing
            device: Device to use ('cuda', 'cpu', or 'auto')
            speaker_model: Pre-loaded speaker verification model (optional)
            asr_model: Pre-loaded ASR model (optional)
            speaker_model_name: NeMo speaker verification model name
            asr_model_name: NeMo ASR model name
            openai_model: OpenAI model name
        """
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        
        # Use provided models or load new ones
        if speaker_model is not None:
            print("Using provided speaker verification model...")
            self.speaker_model = speaker_model
            self.speaker_model.to(self.device)
            self.speaker_model.freeze()
        else:
            print(f"Loading speaker verification model on {self.device}...")
            self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                speaker_model_name
            )
            self.speaker_model.to(self.device)
            self.speaker_model.freeze()
        
        if asr_model is not None:
            print("Using provided ASR model...")
            self.asr_model = asr_model
        else:
            print(f"Loading ASR model for re-transcription...")
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(asr_model_name)
        
        self.client = OpenAI(api_key=openai_api_key)
        self.openai_model = openai_model
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Remove punctuation and lowercase text."""
        return re.sub(r'[^\w\s]', '', text.lower())
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess text by lowercasing, removing punctuation, and extra spaces."""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Strip extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
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
        
        return emb.detach().cpu().numpy().squeeze()
    
    def compute_embedding_similarity(
        self, 
        emb1: np.ndarray, 
        emb2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score [0, 1]
        """
        emb1_tensor = torch.from_numpy(emb1)
        emb2_tensor = torch.from_numpy(emb2)
        
        X = emb1_tensor / torch.linalg.norm(emb1_tensor)
        Y = emb2_tensor / torch.linalg.norm(emb2_tensor)
        
        similarity = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        return ((similarity + 1) / 2).item()
    
    def build_speaker_database(
        self,
        diarization: List[Tuple],
        audio_array: np.ndarray,
        num_speakers: int = 4,
        sr: int = 16000
    ) -> Tuple[faiss.Index, np.ndarray]:
        """Build FAISS database of speaker embeddings.
        
        Args:
            diarization: List of (start, end, speaker) tuples
            audio_array: Full audio array
            num_speakers: Number of speakers
            sr: Sample rate
            
        Returns:
            Tuple of (FAISS index, speaker labels array)
        """
        speaker_segments = {f'Speaker {i}': [] for i in range(num_speakers)}

        # Group audio by speaker
        for seg in diarization:
            start, end, speaker = seg[:3]
            if speaker in speaker_segments:
                segment_audio = audio_array[int(start * sr):int(end * sr)]
                segment_audio = segment_audio / np.max(np.abs(segment_audio) + 1e-8)
                
                try:
                    embed = self.get_embedding(segment_audio)
                    speaker_segments[speaker].append(embed.tolist())
                except Exception:
                    pass
        
        all_embeddings = []
        all_labels = []
        
        for speaker, embeddings in speaker_segments.items():
            if embeddings:
                embeddings_array = np.asarray(embeddings, dtype=np.float32)
                all_embeddings.append(embeddings_array)
                all_labels += [speaker] * len(embeddings_array)
        
        if not all_embeddings:
            raise ValueError("No valid embeddings found")
        
        all_embeddings = np.vstack(all_embeddings)
        faiss_index = faiss.IndexFlatIP(all_embeddings.shape[-1])
        faiss_index.add(all_embeddings)
        
        return faiss_index, np.asarray(all_labels)
    
    def reverify_speakers(
        self,
        diarization: List[Tuple],
        audio_array: np.ndarray,
        faiss_index: faiss.Index,
        speaker_labels: np.ndarray,
        sr: int = 16000,
        num_retrieval: int = 10
    ) -> List[str]:
        """Re-verify speaker labels using FAISS retrieval.
        
        Args:
            diarization: List of segments
            audio_array: Full audio array
            faiss_index: FAISS index of embeddings
            speaker_labels: Array of speaker labels
            sr: Sample rate
            num_retrieval: Number of nearest neighbors to retrieve
            
        Returns:
            List of re-verified speaker labels
        """
        results = []
        
        for seg in diarization:
            start, end = seg[:2]
            segment_audio = audio_array[int(start * sr):int(end * sr)]
            segment_audio = segment_audio / np.max(np.abs(segment_audio) + 1e-8)
            
            try:
                emb = self.get_embedding(segment_audio)
                emb = np.reshape(emb, (1, -1))
                
                _, retrieved_idx = faiss_index.search(emb, num_retrieval)
                retrieved_labels = speaker_labels[retrieved_idx[0]]
                
                counter = Counter(retrieved_labels)
                most_common_label, _ = counter.most_common(1)[0]
                results.append(most_common_label)
            except Exception:
                results.append('Unknown')
        
        return results
    
    def merge_words_to_segments(
        self,
        diarization: List[Tuple],
        transcript: List[Dict]
    ) -> List[List]:
        """Merge word-level transcripts with diarization segments.
        
        Args:
            diarization: List of (start, end, speaker) tuples
            transcript: List of word dictionaries with timestamps
            
        Returns:
            List of [start, end, speaker, text, word_times]
        """
        matched_segments = []
        used_word_indices = set()
        
        for seg_start, seg_end, speaker in diarization:
            seg_words = []
            seg_word_times = []
            
            for idx, word in enumerate(transcript):
                if seg_start <= word['start'] <= seg_end:
                    seg_words.append(word['word'])
                    seg_word_times.append((word['start'], word['end'], word['word']))
                    used_word_indices.add(idx)
            
            sentence = ' '.join(seg_words)
            matched_segments.append([seg_start, seg_end, speaker, sentence, seg_word_times])
        
        # Include unmatched words as their own segments
        for i, word in enumerate(transcript):
            if i not in used_word_indices:
                matched_segments.append([
                    word['start'],
                    word['end'],
                    'Unknown',
                    word['word'],
                    [(word['start'], word['end'], word['word'])]
                ])
        
        matched_segments.sort(key=lambda x: x[0])
        return matched_segments
    
    def merge_segments(
        self,
        segments: List[List],
        max_gap: float = 0.5
    ) -> List[List]:
        """Merge adjacent segments with same speaker and small gaps.
        
        Args:
            segments: List of segments [start, end, speaker, text, word_times]
            max_gap: Maximum gap in seconds to merge
            
        Returns:
            Merged segments
        """
        if not segments:
            return []

        # Start with first segment
        merged = [segments[0]]
        
        for seg in segments[1:]:
            prev_start, prev_end, prev_spk, prev_text, prev_times = merged[-1]
            start, end, spk, text, times = seg
            # Check if speaker is same (Unknown counts as previous) and time gap is small
            same_speaker = (spk == prev_spk) or (spk == 'Unknown') or (prev_spk == 'Unknown')
            prev_no_punct = not prev_text.strip().endswith((',', '.', '!', '?'))
            gap = start - prev_end
            
            if gap <= max_gap and same_speaker and prev_no_punct:
                # Merge: update end time and concatenate word
                merged_times = prev_times + times
                merged[-1] = [
                    prev_start, 
                    end, 
                    prev_spk if prev_spk != 'Unknown' else spk,
                    prev_text + ' ' + text, 
                    merged_times
                ]
            else:
                merged.append(seg)
        
        return merged
    
    def retranscribe_segments(
        self,
        segments: List[List],
        audio_array: np.ndarray,
        sr: int = 16000,
        temp_dir: str = 'tmp_audio'
    ) -> List[List]:
        """Re-transcribe segments with empty or Unknown text.
        
        Args:
            segments: List of segments
            audio_array: Full audio array
            sr: Sample rate
            temp_dir: Temporary directory for audio files
            
        Returns:
            Updated segments with re-transcribed text
        """
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, 'asr_tmp.wav')
        
        updated_segments = []
        
        for seg in tqdm(segments, desc="Re-transcribing"):
            seg_start, seg_end, speaker, text, word_times = seg
            
            # Re-transcribe if text is empty
            if text.replace(" ", "") == '':
                segment_audio = audio_array[int(seg_start * sr):int(seg_end * sr)]
                sf.write(temp_file, segment_audio, sr)
                
                output = self.asr_model.transcribe(
                    temp_file, 
                    batch_size=1, 
                    timestamps=True, 
                    verbose=False
                )
                
                if not output[0].timestamp['word']:
                    continue
                
                # Skip single character or "Mm" utterances
                if (len(output[0].timestamp['word']) == 1 and 
                    (len(output[0].timestamp['word'][0]['word']) == 1 or 
                     output[0].timestamp['word'][0]['word'].startswith('Mm'))):
                    continue
                
                new_word_times = [
                    (w['start'] + seg_start, w['end'] + seg_start, w['word'])
                    for w in output[0].timestamp['word']
                ]
                
                updated_segments.append([
                    seg_start, seg_end, speaker, output[0].text, new_word_times
                ])
                continue
            
            # Verify Unknown segments with re-transcription
            if speaker == 'Unknown':
                segment_audio = audio_array[int(seg_start * sr):int(seg_end * sr)]
                sf.write(temp_file, segment_audio, sr)
                
                output = self.asr_model.transcribe(
                    temp_file, 
                    batch_size=1, 
                    timestamps=True, 
                    verbose=False
                )
                
                lev_sim = textdistance.levenshtein.normalized_similarity(
                    self.preprocess_text(output[0].text),
                    self.preprocess_text(text)
                )
                
                if lev_sim >= 0.9:
                    updated_segments.append(seg)
                continue
            
            updated_segments.append(seg)
        
        return updated_segments
    
    def gpt_infer_speaker(
        self,
        context: Dict,
    ) -> List:
        """Use GPT to infer speaker label based on context.
        
        Args:
            context: Dict with preceding, current, and following segments
            
        Returns:
            GPT prediction [speaker_label, text, probability]
        """
        prompt = """You are given a text segment from a conversation transcript. Each item in the list contains a speaker label, and text. 
Assign the most likely speaker label to the current text segment by considering the semantic context of **both the preceding and following labeled segments.**
Use similarities in meaning, tone, and dialogue flow from the surrounding context to make the best decision.

**Input format:**  
A dict of tuples in the form:  
{"preceding":[[spk_label_1, text_1], [spk_label_2, text_2], ...],
 "current": {"time_gap_previous":<the time (in seconds) between the end of the previous text and the start of the current text.>,
             "time_gap_next":<the time (in seconds) between the end of the current text and the start of the next text.>, 
             "content": "text"
             },
 "following": [[spk_label_1, text_1], [spk_label_2, text_2], ...]
 }
 
Note: 
- Use both the **following** and preceding context **equally** when deciding the speaker for the current text.

**Output format, list of the content:**  
Each item must follow this structure: ["<speaker_label>", <text>, <probability>]
Where:
<speaker_label> = predicted speaker label
<text> = the corresponding text
<probability> = the confidence score for the predicted label

Note:
- Only return the list. Do not include any additional text, explanation, or formatting.
- Use double quotes for all strings.
- Label a speaker as 'Background' if **you are very sure their speech was unintentionally captured**, such as background noise, incidental remarks, or distant voices.
Input:
"""
        gpt_input = prompt + str(context)
        
        completion = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": gpt_input}]
        )
        
        result = completion.choices[0].message.content
        
        try:
            return eval(result)  # Parse list from string
        except Exception:
            return []
    
    def gpt_identify_speakers(self, conversation: str) -> Dict:
        """Use GPT to assign identities to speakers and identify background speakers.
        
        Args:
            conversation: Full conversation text with speaker labels
            
        Returns:
            Dict mapping original speaker labels to [identity, type]
        """
        prompt = """You are given a conversation with speaker labels. Your task is to assign a identity to each speaker.  
    
Return the result in the following format:  
{
    "original_speaker_label": ["assigned_identity", "background or main speaker"],
    ...
}

Notes:
- Mark a speaker as "background" only if you are certain their speech was unintentionally captured (e.g., background noise, incidental talk, or distant voices).
- Mark speakers who actively participate in the conversation as "main".
- If different original speaker labels refer to the same person, merge them by assigning the same identity, but only do this if you are very sure.
- Do not include any additional text, explanation, or formatting.

Input:
"""
        gpt_input = prompt + conversation
        
        completion = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": gpt_input}]
        )
        
        result = completion.choices[0].message.content
        return json.loads(result)
    
    @staticmethod
    def get_speaker_identity_mapping(identity_dict: Dict) -> Dict:
        """Create mapping from original labels to unified identities.
        
        Args:
            identity_dict: Dict from GPT with speaker identities
            
        Returns:
            Mapping of speaker labels to unified identities
        """
        mapping = {}
        seen_values = {}
        
        for speaker, value in identity_dict.items():
            val_tuple = tuple(value)
            if val_tuple in seen_values:
                mapping[speaker] = seen_values[val_tuple]
            else:
                mapping[speaker] = speaker
                seen_values[val_tuple] = speaker
        
        return mapping
    
    @staticmethod
    def choose_majority_speaker(s1: str, s2: str, s3: str) -> str:
        """Choose speaker by majority vote, with preference for s2 on tie.
        
        Args:
            s1: First speaker label
            s2: Second speaker label (GPT prediction)
            s3: Third speaker label
            
        Returns:
            Chosen speaker label
        """
        if s2 == 'Unknown':
            return s1
        
        strings = [s1, s2, s3]
        counts = {s: strings.count(s) for s in strings}
        
        sorted_strings = sorted(
            counts.items(), 
            key=lambda x: (-x[1], strings.index(x[0]))
        )
        
        # If tie, return s2 (GPT prediction)
        if list(counts.values()).count(max(counts.values())) > 1:
            return s2
        
        return sorted_strings[0][0]
    
    def gpt_fix_diarization(
        self,
        segments: List[List],
        reverified_labels: List[str],
        window_size: int = 10
    ) -> Tuple[List[List], List]:
        """Use GPT to predict speaker labels based on context.
        
        This is the first step that gets GPT predictions for uncertain segments.
        
        Args:
            segments: List of segments with text
            reverified_labels: Re-verified speaker labels
            window_size: Context window for GPT
            
        Returns:
            Tuple of (segments with updated speakers, GPT outputs)
        """
        new_segments = []
        gpt_outputs = []
        
        for i, seg in enumerate(tqdm(segments, desc="Getting GPT predictions")):
            start, end, speaker, text, word_times = seg
            reverified = reverified_labels[i]
            
            # If confident match, keep original and skip GPT
            if speaker != 'Unknown' and speaker == reverified:
                new_segments.append([start, end, speaker, text])
                gpt_outputs.append([])
                continue
            
            # Build context for GPT
            prev_context = [
                [s[2], s[3]] for s in new_segments[-window_size:]
            ]
            
            next_context = [
                [s[2], s[3]] for s in segments[i+1:i+1+window_size]
            ]
            
            # Calculate time gaps
            gap_previous = round(start - new_segments[-1][1], 3) if new_segments else 0
            gap_next = round(segments[i+1][0] - end, 3) if i+1 < len(segments) else 0
            
            # Split current text into sentences
            current_content = [
                ['Unknown', s.strip()] 
                for s in re.split(r'[.?]', text) if s.strip()
            ]
            
            context = {
                "preceding": prev_context,
                "current": {
                    "time_gap_previous": gap_previous,
                    "time_gap_next": gap_next,
                    "content": current_content
                },
                "following": next_context
            }
            
            # Get GPT prediction
            gpt_result = self.gpt_infer_speaker(context)
            
            # Store GPT result
            if gpt_result and not isinstance(gpt_result[0], list):
                gpt_result = [gpt_result]
            gpt_outputs.append(gpt_result)
            
            # For now, use GPT prediction if available, otherwise keep original
            if gpt_result and len(gpt_result) > 0:
                final_speaker = gpt_result[0][0]
            else:
                final_speaker = speaker
            
            new_segments.append([start, end, final_speaker, text])
        
        return new_segments, gpt_outputs
    
    def combine_predictions(
        self,
        segments: List[List],
        gpt_outputs: List,
        reverified_labels: List[str],
        speaker_mapping: Dict[str, str],
        confidence_threshold: float = 0.9
    ) -> List[List]:
        """Combine original labels, GPT predictions, and re-verified labels.
        
        This is the second step that intelligently combines all predictions.
        
        Args:
            segments: Segments with GPT-predicted speakers
            gpt_outputs: GPT output predictions with confidence scores
            reverified_labels: Re-verified speaker labels from embeddings
            speaker_mapping: Identity mapping from get_speaker_identity_mapping
            confidence_threshold: Confidence threshold for GPT predictions
            
        Returns:
            Final refined segments
        """
        final_segments = []
        
        for i, (seg, gpt_result, reverified) in enumerate(
            zip(segments, gpt_outputs, reverified_labels)
        ):
            start, end, original_speaker, text = seg[:4]
            
            try:
                # Handle GPT formatting issues
                if gpt_result and not isinstance(gpt_result[0], list):
                    gpt_result = [gpt_result]
                
                # Skip if GPT is very confident it's background
                if (gpt_result and len(gpt_result) > 0 and 
                    gpt_result[0][-1] > confidence_threshold and 
                    gpt_result[0][0] == 'Background'):
                    continue
                
                # Case 1: Original speaker is Unknown
                if original_speaker == 'Unknown':
                    # Skip if GPT has low confidence or predicts Unknown
                    if (not gpt_result or len(gpt_result) == 0 or
                        gpt_result[0][-1] < confidence_threshold or 
                        gpt_result[0][0] == 'Unknown'):
                        continue
                    
                    # Use GPT prediction with identity mapping
                    final_speaker = speaker_mapping.get(
                        gpt_result[0][0], 
                        gpt_result[0][0]
                    )
                    final_segments.append([start, end, final_speaker, text])
                    continue
                
                # Case 2: Original matches re-verified OR GPT has low confidence
                if (original_speaker == reverified or 
                    not gpt_result or len(gpt_result) == 0 or
                    gpt_result[0][-1] < confidence_threshold):
                    # Use original with identity mapping
                    final_speaker = speaker_mapping.get(
                        original_speaker, 
                        original_speaker
                    )
                    final_segments.append([start, end, final_speaker, text])
                    continue
                
                # Case 3: Disagreement - use majority vote
                gpt_speaker = speaker_mapping.get(
                    gpt_result[0][0], 
                    gpt_result[0][0]
                )
                original_mapped = speaker_mapping.get(
                    original_speaker, 
                    original_speaker
                )
                reverified_mapped = speaker_mapping.get(
                    reverified, 
                    reverified
                )
                
                final_speaker = self.choose_majority_speaker(
                    original_mapped, 
                    gpt_speaker, 
                    reverified_mapped
                )
                final_segments.append([start, end, final_speaker, text])
                
            except Exception as e:
                # On error, use original speaker with mapping
                print(f"Error processing segment {i}: {e}")
                if original_speaker != 'Unknown':
                    final_speaker = speaker_mapping.get(
                        original_speaker, 
                        original_speaker
                    )
                    final_segments.append([start, end, final_speaker, text])
        
        return final_segments
    
    def refine_with_gpt(
        self,
        segments: List[List],
        reverified_labels: List[str],
        audio_array: np.ndarray,
        sr: int = 16000,
        window_size: int = 10,
        confidence_threshold: float = 0.9
    ) -> Tuple[List[List], List]:
        """Refine diarization using GPT and speaker re-verification.
        
        Args:
            segments: List of segments with text
            reverified_labels: Re-verified speaker labels
            audio_array: Full audio array
            sr: Sample rate
            window_size: Context window for GPT
            confidence_threshold: Confidence threshold for GPT predictions
            
        Returns:
            Tuple of (refined segments, GPT outputs)
        """
        # Step 1: Get speaker identities from full conversation
        conversation_text = "\n".join([
            f"{seg[2]}: {seg[3]}" for seg in segments
        ])
        
        print("Getting speaker identities...")
        identity_dict = self.gpt_identify_speakers(conversation_text)
        speaker_mapping = self.get_speaker_identity_mapping(identity_dict)
        
        # Step 2: Get GPT predictions for uncertain segments
        print("Getting GPT predictions...")
        _, gpt_outputs = self.gpt_fix_diarization(
            segments, 
            reverified_labels, 
            window_size
        )
        
        # Step 3: Combine all predictions intelligently
        print("Combining predictions...")
        final_segments = self.combine_predictions(
            segments, # Original segments
            gpt_outputs,
            reverified_labels,
            speaker_mapping,
            confidence_threshold
        )
        
        return final_segments, gpt_outputs
    
    @staticmethod
    def remove_duplicate_segments(segments: List[List]) -> List[List]:
        """Remove duplicate segments (same speaker, overlapping time, duplicate text).
        
        Args:
            segments: List of segments
            
        Returns:
            Cleaned segments
        """
        keep = [True] * len(segments)
        
        for i, seg in enumerate(segments):
            start, end, speaker, text = seg[:4]
            text_norm = DiarizationRefiner.normalize_text(text)
            
            for j in range(i):
                prev_start, prev_end, prev_speaker, prev_text = segments[j][:4]
                prev_text_norm = DiarizationRefiner.normalize_text(prev_text)
                
                # Check if current is contained in previous
                if (start >= prev_start and end <= prev_end and 
                    speaker == prev_speaker):
                    # Check if all words in current exist in previous
                    if all(w in prev_text_norm.split() for w in text_norm.split()):
                        keep[i] = False
                        break
        
        return [seg for seg, k in zip(segments, keep) if k]
    
    def process(
        self,
        diarization: List[Tuple],
        transcript: List[Dict],
        audio_path: str,
        num_speakers: int = 4,
        sample_rate: int = 16000,
        enable_gpt: bool = True,
        enable_retranscribe: bool = True,
        gpt_window_size: int = 10,
        gpt_confidence_threshold: float = 0.9,
        num_retrieval: int = 10
    ) -> List[List]:
        """Complete refinement pipeline.
        
        Args:
            diarization: Raw diarization output
            transcript: Word-level transcript
            audio_path: Path to audio file
            num_speakers: Number of speakers
            sample_rate: Audio sample rate
            enable_gpt: Whether to use GPT refinement
            enable_retranscribe: Whether to re-transcribe segments
            gpt_window_size: Context window for GPT
            gpt_confidence_threshold: Confidence threshold for GPT
            num_retrieval: Number of neighbors for speaker verification
            
        Returns:
            Refined diarization segments
        """
        print("Loading audio...")
        audio_array, _ = load_audio_ffmpeg(audio_path, sr=sample_rate)
        
        print("Merging transcripts with diarization...")
        merged = self.merge_words_to_segments(diarization, transcript)
        
        print("Merging adjacent segments (initial)...")
        merged = self.merge_segments(merged, max_gap=0.01)
        
        if enable_retranscribe:
            print("Re-transcribing segments...")
            merged = self.retranscribe_segments(merged, audio_array, sample_rate)
            print("Merging adjacent segments (post re-transcription)...")
            merged = self.merge_segments(merged, max_gap=0.25)

        print("Building speaker database...")
        faiss_index, speaker_labels = self.build_speaker_database(
            diarization, audio_array, num_speakers, sample_rate
        )

        print("Re-verifying speakers...")
        reverified = self.reverify_speakers(
            merged, audio_array, faiss_index, speaker_labels,
            sample_rate, num_retrieval
        )
        
        if enable_gpt:
            print("Refining with GPT...")
            refined, _ = self.refine_with_gpt(
                merged, reverified, audio_array, sample_rate,
                gpt_window_size, gpt_confidence_threshold
            )
        else:
            # Without GPT, build database and re-verify only
            # Use re-verified labels directly
            refined = [[s[0], s[1], reverified[i], s[3]] for i, s in enumerate(merged)]
        
        print("Cleaning duplicates...")
        final = self.remove_duplicate_segments(refined)
        
        return final