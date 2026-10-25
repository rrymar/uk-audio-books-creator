# Ukrainian audiobooks from Russian text

Translate a Russian `.txt` book to Ukrainian and turn it into an audiobook with [StyleTTS2 Ukrainian](https://huggingface.co/spaces/patriotyk/styletts2-ukrainian). Everything runs locally.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (for `.mp3` output).

```
brew install ffmpeg
uv sync
```

Models download on first run to `~/.cache/huggingface/hub/` (about 2.2 GB).

## 1. Translate Russian → Ukrainian

Uses the [Helsinki-NLP/opus-mt-ru-uk](https://huggingface.co/Helsinki-NLP/opus-mt-ru-uk) model.

```
.venv/bin/python translate_ru_uk.py book.txt                  # writes book_uk.txt
.venv/bin/python translate_ru_uk.py book.txt -o out.txt
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | `<input>_uk.txt` | Output file |
| `--batch-size` | `16` | Sentences translated at once |

## 2. Ukrainian text → audio

Use the parallel script for whole books. It runs about twice as fast as the single-process script.

```
.venv/bin/python txt_to_audio_styletts_uk_parallel.py book_uk.txt                     # writes book_uk.mp3
.venv/bin/python txt_to_audio_styletts_uk_parallel.py book_uk.txt --workers 8
.venv/bin/python txt_to_audio_styletts_uk_parallel.py book_uk.txt --voice "Олександр Ролдугін"
```

Single process, handy for short tests:

```
.venv/bin/python txt_to_audio_styletts_uk.py sample_uk.txt -o sample_uk_styletts.mp3
```

List the available voices:

```
.venv/bin/python txt_to_audio_styletts_uk.py --list-voices
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | `<input>.mp3` | Output `.mp3` or `.wav` |
| `--voice` | Filatov (single-speaker model) | Multi-speaker voice name from `--list-voices` |
| `--speed` | `1.0` | Speech speed, `0.7`–`1.3` |
| `--workers` | `4` | CPU worker processes (parallel script only) |
| `--gpu-workers` | `0` | Extra workers on the Mac GPU (parallel script only, untested) |
| `--device` | `cpu` | `cpu` or `mps` (single-process script only; CPU is faster for this model) |

## Full pipeline

```
.venv/bin/python translate_ru_uk.py book.txt
.venv/bin/python txt_to_audio_styletts_uk_parallel.py book_uk.txt --voice "Олександр Ролдугін"
```

## Notes

- Neither script resumes: an interrupted run starts over.
- Footnote markers like `[1]` are removed before speech.
- Numbers and abbreviations are not converted to words and may be skipped or misread; spell them out in the text for best results.
- Machine translation is decent but literal; slang and idioms may come out wrong.
