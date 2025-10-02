"""
Complete Audio Diarization and Transcription Pipeline
Supports diarization, transcription, and LLM-based refinement
"""

import json
import argparse
from pathlib import Path

from pydub import AudioSegment

from src.asr import AudioTranscriber
from src.diarization import AudioDiarizer
from src.refinement import DiarizationRefiner
from src.utils import create_directory


def main():
    parser = argparse.ArgumentParser(
        description="Audio Diarization, Transcription, and Refinement Pipeline"
    )
    
    # Required arguments
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["diarize", "transcribe", "refine", "full"],
        default="full",
        help="Processing mode: diarize, transcribe, refine, or full (default: full)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    # Diarization options
    parser.add_argument(
        "--n-speakers",
        type=int,
        default=4,
        help="Expected number of speakers (default: 4)"
    )
    
    # Processing options
    parser.add_argument(
        "--snippet-duration",
        type=int,
        default=250,
        help="Snippet duration in seconds (default: 250)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding around audio snippets in seconds (default: 5)"
    )
    
    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: cuda, cpu, or auto (default: auto)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate for audio processing (default: inferred from file)"
    )
    
    # Refinement options
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key for GPT-based refinement (required for refine/full mode)"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model to use (default: gpt-4.1)"
    )
    parser.add_argument(
        "--enable-gpt",
        action="store_true",
        default=True,
        help="Enable GPT-based refinement (default: True)"
    )
    parser.add_argument(
        "--disable-gpt",
        action="store_false",
        dest="enable_gpt",
        help="Disable GPT-based refinement"
    )
    parser.add_argument(
        "--enable-retranscribe",
        action="store_true",
        default=True,
        help="Enable re-transcription of uncertain segments (default: True)"
    )
    parser.add_argument(
        "--disable-retranscribe",
        action="store_false",
        dest="enable_retranscribe",
        help="Disable re-transcription"
    )
    parser.add_argument(
        "--gpt-window-size",
        type=int,
        default=10,
        help="Context window size for GPT refinement (default: 10)"
    )
    parser.add_argument(
        "--gpt-confidence-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for GPT predictions (default: 0.9)"
    )
    parser.add_argument(
        "--num-retrieval",
        type=int,
        default=10,
        help="Number of neighbors for speaker re-verification (default: 10)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ["refine", "full"] and args.enable_gpt and not args.openai_api_key:
        parser.error("--openai-api-key is required when using refine or full mode with GPT enabled")

    # Create output directory
    create_directory(args.output_dir)
    output_name = Path(args.audio_path).stem

    # Infer sampling rate from audio if not provided
    if args.sample_rate is None:
        sample_rate = AudioSegment.from_file(args.audio_path).frame_rate
        print(f"Inferred sample rate: {sample_rate} Hz")
    else:
        sample_rate = args.sample_rate

    # Initialize models that will be reused
    speaker_model = None
    asr_model = None
    diarizer = None

    # Step 1: Diarization
    segments = None
    if args.mode in ["diarize", "full"]:
        print("\n" + "="*50)
        print("STEP 1: DIARIZATION")
        print("="*50)
        
        diarizer = AudioDiarizer(device=args.device, sample_rate=sample_rate)
        segments = diarizer.process_audio(
            args.audio_path,
            snippet_duration=args.snippet_duration,
            padding=args.padding,
            n_speakers=args.n_speakers
        )

        output_path = f"{args.output_dir}/{output_name}_diarization.json"
        with open(output_path, 'w') as f:
            json.dump(segments, f, indent=2)
        print(f"\n✓ Diarization saved to: {output_path}")
        print(f"  Found {len(segments)} segments")
        
        # Extract speaker model for reuse (if refinement will be done)
        if args.mode == "full":
            print("\nExtracting speaker model for refinement stage...")
            speaker_model = diarizer.speaker_model
            
            # Clear diarization model from memory
            print("Clearing diarization model from memory...")
            del diarizer.diar_model
            import gc
            gc.collect()
            if args.device in ["cuda", "auto"]:
                import torch
                torch.cuda.empty_cache()

    # Step 2: Transcription
    words = None
    if args.mode in ["transcribe", "full"]:
        print("\n" + "="*50)
        print("STEP 2: TRANSCRIPTION")
        print("="*50)
        
        transcriber = AudioTranscriber()
        words = transcriber.transcribe_audio(
            args.audio_path,
            snippet_duration=args.snippet_duration,
            padding=args.padding
        )

        output_path = f"{args.output_dir}/{output_name}_transcription.json"
        with open(output_path, 'w') as f:
            json.dump(words, f, indent=2)
        print(f"\n✓ Transcription saved to: {output_path}")
        print(f"  Transcribed {len(words)} words")
        
        # Extract ASR model for reuse (if refinement will be done)
        if args.mode == "full":
            print("\nExtracting ASR model for refinement stage...")
            asr_model = transcriber.model

    # Step 3: Refinement
    if args.mode in ["refine", "full"]:
        print("\n" + "="*50)
        print("STEP 3: REFINEMENT")
        print("="*50)
        
        # Load diarization and transcription if not already done
        if segments is None:
            diar_path = f"{args.output_dir}/{output_name}_diarization.json"
            print(f"Loading diarization from: {diar_path}")
            with open(diar_path, 'r') as f:
                segments = json.load(f)
        
        if words is None:
            trans_path = f"{args.output_dir}/{output_name}_transcription.json"
            print(f"Loading transcription from: {trans_path}")
            with open(trans_path, 'r') as f:
                words = json.load(f)
        
        # Create refiner with pre-loaded models if available
        refiner = DiarizationRefiner(
            openai_api_key=args.openai_api_key,
            device=args.device,
            speaker_model=speaker_model,
            asr_model=asr_model,
            openai_model=args.openai_model
        )
        
        refined_segments = refiner.process(
            diarization=segments,
            transcript=words,
            audio_path=args.audio_path,
            num_speakers=args.n_speakers,
            sample_rate=sample_rate,
            enable_gpt=args.enable_gpt,
            enable_retranscribe=args.enable_retranscribe,
            gpt_window_size=args.gpt_window_size,
            gpt_confidence_threshold=args.gpt_confidence_threshold,
            num_retrieval=args.num_retrieval
        )

        output_path = f"{args.output_dir}/{output_name}_refined.json"
        with open(output_path, 'w') as f:
            json.dump(refined_segments, f, indent=2)
        print(f"\n✓ Refined diarization saved to: {output_path}")
        print(f"  Final segments: {len(refined_segments)}")

    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()