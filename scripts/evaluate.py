from __future__ import annotations

import argparse
from pathlib import Path

from backend.evaluator import evaluate_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated audiobook audio against a reference voice.")
    parser.add_argument("--reference", type=str, required=True, help="Reference audio WAV/MP3 path")
    parser.add_argument("--audio", type=str, required=True, help="Generated audio WAV/MP3 path")
    parser.add_argument("--chunks", type=int, default=0, help="Total chunks generated")
    args = parser.parse_args()

    result = evaluate_audio(args.reference, args.audio, chunk_count=args.chunks)
    print(result)


if __name__ == "__main__":
    main()
