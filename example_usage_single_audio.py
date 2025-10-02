import json
import argparse
from pathlib import Path

from pydub import AudioSegment

from src.asr import AudioTranscriber
from src.diarization import AudioDiarizer

from src.utils import (
    create_directory,
)


def main():
    parser = argparse.ArgumentParser(
        description="Audio Diarization and Transcription Pipeline"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["diarize", "transcribe", "both"],
        default="both",
        help="Processing mode"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--n-speakers",
        type=int,
        default=4,
        help="Expected number of speakers (for diarization)"
    )
    parser.add_argument(
        "--snippet-duration",
        type=int,
        default=250,
        help="Snippet duration in seconds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,   # None means "not provided"
        help="Sample rate for audio processing (if not provided, inferred from file)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding around audio snippets in seconds"
    )

    args = parser.parse_args()

    # Create output directory
    create_directory(args.output_dir)
    output_name = Path(args.audio_path).stem

    # Infer sampling rate from audio if not provided
    if args.sample_rate is None:
        sample_rate = AudioSegment.from_file(args.audio_path).frame_rate
        print(f"Inferred sample rate {sample_rate} Hz from {args.audio_path}")
    else:
        sample_rate = args.sample_rate

    # Process based on mode
    if args.mode in ["diarize", "both"]:
        print("\n=== Starting Diarization ===")
        diarizer = AudioDiarizer(device=args.device, sample_rate=sample_rate)
        segments = diarizer.process_audio(
            args.audio_path,
            snippet_duration=args.snippet_duration,
            padding=args.padding,
            n_speakers=args.n_speakers
        )

        # Save results
        output_path = f"{args.output_dir}/{output_name}_diarization.json"
        with open(output_path, 'w') as f:
            json.dump(segments, f, indent=2)
        print(f"Diarization saved to: {output_path}")

    if args.mode in ["transcribe", "both"]:
        print("\n=== Starting Transcription ===")
        transcriber = AudioTranscriber()
        words = transcriber.transcribe_audio(
            args.audio_path,
            snippet_duration=args.snippet_duration,
            padding=args.padding
        )

        # Save results
        output_path = f"{args.output_dir}/{output_name}_transcription.json"
        with open(output_path, 'w') as f:
            json.dump(words, f, indent=2)
        print(f"Transcription saved to: {output_path}")

    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()