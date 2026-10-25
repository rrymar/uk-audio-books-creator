import argparse
import re
import sys
import time
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

from txt_to_audio_styletts_uk import format_time

MODEL_NAME = "Helsinki-NLP/opus-mt-ru-uk"


def split_sentences(line: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?…])\s+", line.strip()) if s]


def show_progress(done: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    ratio = done / total
    eta = format_time(elapsed * (total - done) / done) if done else "--:--:--"
    bar = "█" * int(ratio * 30) + "░" * (30 - int(ratio * 30))
    print(f"\r{bar} {ratio:6.1%} {done}/{total} elapsed {format_time(elapsed)} eta {eta}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate a Russian .txt file to Ukrainian with opus-mt-ru-uk")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Output .txt (default: <input>_uk.txt)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    line_sentences = [split_sentences(line) for line in lines]
    sentences = [s for group in line_sentences for s in group]
    if not sentences:
        sys.exit(f"No text found in {args.input}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device).eval()

    translated = []
    started = time.monotonic()
    show_progress(0, len(sentences), started)
    for start in range(0, len(sentences), args.batch_size):
        batch = sentences[start:start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, num_beams=4)
        translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        show_progress(len(translated), len(sentences), started)
    print()

    output = args.output or args.input.with_name(f"{args.input.stem}_uk.txt")
    result = iter(translated)
    output_lines = [" ".join(next(result) for _ in group) for group in line_sentences]
    output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
