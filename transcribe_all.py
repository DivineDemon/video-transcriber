#!/usr/bin/env python3
"""
Auto-transcribe all videos in the videos/ directory.

Uses MLX Whisper (Apple Silicon GPU) with the large-v3-turbo model, which runs
at ~10x real-time on an M-series Mac while keeping near large-v3 accuracy.

Pipeline:
  1. scan videos/ subdirectories, skipping ones already transcribed
     (unless --force)
  2. extract mono 16kHz audio from all pending videos in parallel with ffmpeg
  3. transcribe sequentially on the GPU, writing <name>-transcript.txt and .json
  4. sanity-check that each transcript covers the full audio duration
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlx_whisper import transcribe as mlx_transcribe

VIDEOS_DIR = Path(__file__).resolve().parent / "videos"
MODEL = "mlx-community/whisper-large-v3-turbo"  # ~10x real-time on M-series GPU
LANGUAGE = "en"  # set to None to auto-detect
WORD_TIMESTAMPS = (
    False  # True doubles runtime; only enable if you need words in the .json
)
VIDEO_EXTS = (".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v")


def ffprobe_duration(path: Path) -> float:
    """Duration in seconds, or 0 if probe fails."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return float(out.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        return 0.0


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract mono 16kHz PCM audio; returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[-500:]}", file=sys.stderr)
        return False
    return True


def transcribe(audio_path: Path, duration: float) -> dict:
    """Transcribe one audio file into segments. End times are clipped to the
    real audio duration (whisper often hallucinates a few seconds past the
    end), and empty segments are dropped."""
    result = mlx_transcribe(
        str(audio_path),
        path_or_hf_repo=MODEL,
        language=LANGUAGE,
        word_timestamps=WORD_TIMESTAMPS,
    )
    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        start, end = seg["start"], seg["end"]
        if duration > 0:
            start = max(0.0, start)
            end = min(end, duration)
            if start >= end:
                start = max(0.0, min(start, duration - 0.01))
                end = duration
        segments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
            }
        )
    return {"segments": segments}


def is_done(video_dir: Path) -> bool:
    return (video_dir / f"{video_dir.name}-transcript.txt").exists() and (
        video_dir / f"{video_dir.name}-transcript.json"
    ).exists()


def find_video_file(video_dir: Path) -> Path | None:
    files = [f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]
    if not files:
        return None
    return max(
        files, key=lambda f: f.stat().st_size
    )  # biggest file = the real recording


def write_outputs(segments: list, txt_path: Path, json_path: Path):
    with open(json_path, "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w") as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")


def coverage_warning(segments: list, duration: float) -> str | None:
    """Alert if the transcript doesn't span the audio (guards against truncated
    or VAD-dropped output, e.g. a transcript that only starts at 190s)."""
    if not segments:
        return "no speech was detected at all"
    if duration <= 0:
        return None
    first, last = segments[0]["start"], segments[-1]["end"]
    problems = []
    if first > max(60.0, duration * 0.10):
        problems.append(
            f"first speech starts at {first:.0f}s but audio is {duration:.0f}s"
        )
    if last < duration * 0.90:
        problems.append(f"last speech ends at {last:.0f}s but audio is {duration:.0f}s")
    return "; ".join(problems) if problems else None


def main(force: bool = False):
    if not VIDEOS_DIR.exists():
        print(f"Videos directory not found: {VIDEOS_DIR}")
        sys.exit(1)

    video_dirs = sorted(d for d in VIDEOS_DIR.iterdir() if d.is_dir())
    if not video_dirs:
        print("No video directories found.")
        return

    jobs = []
    for d in video_dirs:
        pending = force or not is_done(d)
        print(f"  - {d.name} [{'PENDING' if pending else 'done'}]")
        if pending:
            video_file = find_video_file(d)
            if not video_file:
                print(f"[{d.name}] no video file found, skipping", file=sys.stderr)
                continue
            jobs.append((d, video_file))

    if not jobs:
        print("\nNothing to do.")
        return

    print(f"\nModel: {MODEL}  Language: {LANGUAGE or 'auto'}")

    # Stage 1: extract audio from all pending videos in parallel (ffmpeg is
    # I/O + CPU bound and does not touch the GPU, so run them together).
    print("\nExtracting audio...")
    durations: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for video_dir, video_file in jobs:
            audio_path = video_dir / f"{video_dir.name}.wav"
            audio_path.unlink(missing_ok=True)
            durations[video_dir.name] = ffprobe_duration(video_file)
            futures[pool.submit(extract_audio, video_file, audio_path)] = (
                video_dir,
                audio_path,
            )
        for future, (video_dir, audio_path) in futures.items():
            if not future.result():
                print(f"[{video_dir.name}] audio extraction failed", file=sys.stderr)
            else:
                dur = durations[video_dir.name]
                print(f"[{video_dir.name}] extracted ({dur / 60:.1f} min)")

    # Stage 2: transcribe sequentially — the GPU is the bottleneck from here on.
    print("\nTranscribing...")
    done, failed = 0, 0
    for video_dir, _ in jobs:
        name = video_dir.name
        audio_path = video_dir / f"{name}.wav"
        if not audio_path.exists():
            failed += 1
            continue
        duration = durations.get(name, 0.0)
        try:
            t0 = time.time()
            segments = transcribe(audio_path, duration)["segments"]
            elapsed = time.time() - t0
            write_outputs(
                segments,
                video_dir / f"{name}-transcript.txt",
                video_dir / f"{name}-transcript.json",
            )
            done += 1
            rt = duration / max(elapsed, 0.01)
            print(
                f"[{name}] {len(segments)} segments written in {elapsed:.0f}s "
                f"({rt:.1f}x real-time)"
            )
            issue = coverage_warning(segments, duration)
            if issue:
                print(
                    f"[{name}] WARNING: transcript may be incomplete: {issue}",
                    file=sys.stderr,
                )
        except Exception as e:
            failed += 1
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
        finally:
            audio_path.unlink(missing_ok=True)

    print(f"\nAll done! {done} newly transcribed, {failed} failed.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
