import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ukrainian_word_stress import Stressifier

from txt_to_audio_styletts_uk import SAMPLE_RATE, list_voices, load_model, load_parts, prepare, show_progress

worker = {}


def init_worker(voice: str | None, speed: float, threads: int, devices: mp.Queue) -> None:
    device = devices.get()
    torch.set_num_threads(1 if device == "mps" else threads)
    worker["model"], worker["style"] = load_model(voice, device)
    worker["stressify"] = Stressifier()
    worker["speed"] = speed


def synthesize(part: str) -> np.ndarray:
    phonemes = prepare(part, worker["stressify"])
    if not phonemes:
        return np.zeros(0, dtype=np.float32)
    model = worker["model"]
    with torch.no_grad():
        wav = model(model.tokenizer.encode(phonemes), speed=worker["speed"], s_prev=worker["style"])
    return wav.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Ukrainian .txt file to audio with StyleTTS2 using parallel workers")
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("-o", "--output", type=Path, help="Output .wav or .mp3 (default: <input>.mp3)")
    parser.add_argument("--voice", help="Multi-speaker voice name (default: single-speaker Filatov model)")
    parser.add_argument("--speed", type=float, default=1.0, help="0.7 - 1.3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-workers", type=int, default=0, help="Extra workers running on the Mac GPU (mps)")
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

    if args.workers + args.gpu_workers < 1:
        parser.error("need at least one worker")
    threads = max(1, (os.cpu_count() or 4) // max(1, args.workers))
    print(f"Loading {args.workers} CPU workers ({threads} threads each), {args.gpu_workers} GPU workers", flush=True)
    devices = mp.Queue()
    for device in ["mps"] * args.gpu_workers + ["cpu"] * args.workers:
        devices.put(device)

    output = args.output or args.input.with_suffix(".mp3")
    wav_path = output.with_suffix(".wav")
    pool_size = args.workers + args.gpu_workers
    with mp.Pool(pool_size, initializer=init_worker, initargs=(args.voice, args.speed, threads, devices)) as pool:
        started = time.monotonic()
        show_progress(0, len(parts), started)
        with sf.SoundFile(wav_path, "w", samplerate=SAMPLE_RATE, channels=1) as out:
            for i, wav in enumerate(pool.imap(synthesize, parts), 1):
                out.write(wav)
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
