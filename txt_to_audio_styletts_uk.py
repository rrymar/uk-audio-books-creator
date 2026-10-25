import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from unicodedata import normalize

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from ipa_uk import ipa
from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier, StressSymbol

SPACE_REPO = "patriotyk/styletts2-ukrainian"
SINGLE_MODEL = "patriotyk/styletts2_ukrainian_single"
MULTI_MODEL = "patriotyk/styletts2_ukrainian_multispeaker"
SAMPLE_RATE = 24000


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def split_to_parts(text: str) -> list[str]:
    parts = [""]
    last = len(text) - 1
    for i, s in enumerate(text):
        parts[-1] += s
        if s not in ".?!:" or i == last or text[i + 1] != " ":
            continue
        if len(parts[-1]) <= 20:
            continue
        parts.append("")
    return [p.strip() for p in parts if p.strip()]


def load_parts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\[\d+\]", "", text)
    parts = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not re.search(r"\w", line):
            continue
        parts.extend(split_to_parts(line))
    return parts


def prepare(text: str, stressify: Stressifier) -> str:
    text = text.replace('"', "").replace("+", StressSymbol.CombiningAcuteAccent)
    text = normalize("NFKC", text)
    text = re.sub(r"[᠆‐‑‒–—―⁻₋−⸺⸻]", "-", text)
    if text[-1] not in ".?!:-":
        text += "."
    text = re.sub(r" - ", ": ", text)
    return ipa(stressify(text))


def list_voices() -> list[str]:
    files = list_repo_files(SPACE_REPO, repo_type="space")
    return sorted(Path(f).stem for f in files if f.startswith("voices/"))


def load_model(voice: str | None, device: str) -> tuple[StyleTTS2, torch.Tensor]:
    if not voice:
        style_path = hf_hub_download(SPACE_REPO, "filatov.pt", repo_type="space")
        return StyleTTS2(hf_path=SINGLE_MODEL, device=device), torch.load(style_path, map_location=device)

    style_path = hf_hub_download(SPACE_REPO, f"voices/{voice}.pt", repo_type="space")
    return StyleTTS2(hf_path=MULTI_MODEL, device=device), torch.load(style_path, map_location=device)


def show_progress(done: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    ratio = done / total
    eta = format_time(elapsed * (total - done) / done) if done else "--:--:--"
    bar = "█" * int(ratio * 30) + "░" * (30 - int(ratio * 30))
    print(f"\r{bar} {ratio:6.1%} {done}/{total} elapsed {format_time(elapsed)} eta {eta}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Ukrainian .txt file to audio with StyleTTS2")
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("-o", "--output", type=Path, help="Output .wav or .mp3 (default: <input>.mp3)")
    parser.add_argument("--voice", help="Multi-speaker voice name (default: single-speaker Filatov model)")
    parser.add_argument("--speed", type=float, default=1.0, help="0.7 - 1.3")
    parser.add_argument("--device", default="cpu", help="cpu or mps (cpu is faster for this model)")
    parser.add_argument("--list-voices", action="store_true")
    args = parser.parse_args()

    if args.list_voices:
        print("\n".join(list_voices()))
        return
    if not args.input:
        parser.error("input is required")

    parts = load_parts(args.input)
    if not parts:
        sys.exit(f"No text found in {args.input}")

    model, style = load_model(args.voice, args.device)
    stressify = Stressifier()

    output = args.output or args.input.with_suffix(".mp3")
    wav_path = output.with_suffix(".wav")
    started = time.monotonic()
    show_progress(0, len(parts), started)
    with sf.SoundFile(wav_path, "w", samplerate=SAMPLE_RATE, channels=1) as out:
        for i, part in enumerate(parts, 1):
            phonemes = prepare(part, stressify)
            if phonemes:
                with torch.no_grad():
                    wav = model(model.tokenizer.encode(phonemes), speed=args.speed, s_prev=style)
                out.write(wav.cpu().numpy())
            show_progress(i, len(parts), started)
    print()

    if output.suffix.lower() == ".mp3":
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-b:a", "96k", str(output)],
            check=True,
        )
        wav_path.unlink()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
