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
  5. with --delete-videos, delete the source .mov/.mp4 files whose transcripts
     passed that check (irreversible — the transcript is all that remains)

Usage: python transcribe_all.py [--force] [--delete-videos]
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from mlx_whisper import transcribe as mlx_transcribe
from mlx_whisper.audio import load_audio
from faster_whisper.vad import (
    get_speech_timestamps,
    SpeechTimestampsMap,
    VadOptions,
)

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
    """Transcribe one audio file into segments.

    Silero VAD first finds the speech regions; MLX Whisper then transcribes
    only those regions, and segment timestamps are mapped back onto the real
    timeline. This stops whisper from looping hallucinations during long
    stretches of silence (e.g. "Thank you." for minutes, "Bye." forever).

    Empty segments are dropped, exact duplicates are collapsed, and times are
    clipped to the real audio duration. Returns the final speech region end
    alongside the segments.
    """
    audio = load_audio(str(audio_path))
    speech_regions = get_speech_timestamps(audio, VadOptions(), sampling_rate=16000)

    if speech_regions:
        speech_audio = np.concatenate(
            [audio[int(r["start"]) : int(r["end"])] for r in speech_regions]
        )
        map_to_original = SpeechTimestampsMap(speech_regions, 16000)
    else:
        speech_audio = audio
        map_to_original = None

    result = mlx_transcribe(
        speech_audio,
        path_or_hf_repo=MODEL,
        language=LANGUAGE,
        word_timestamps=WORD_TIMESTAMPS,
    )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        if map_to_original is not None:
            # location of this region within the concatenated-speech timeline
            # -> original wall-clock time of the recording
            start = map_to_original.get_original_time(seg["start"])
            end = map_to_original.get_original_time(seg["end"])
        else:
            start, end = seg["start"], seg["end"]
        if duration > 0:
            start = max(0.0, start)
            end = min(end, duration)
            if start >= end:
                start = max(0.0, min(start, duration - 0.01))
                end = duration
        start, end = round(start, 2), round(end, 2)
        if segments and segments[-1] == {"start": start, "end": end, "text": text}:
            continue  # exact duplicate from a hallucination loop
        segments.append({"start": start, "end": end, "text": text})
    # speech_regions values are sample indexes on the ORIGINAL timeline
    last_speech_end = (
        round(speech_regions[-1]["end"] / 16000, 2) if speech_regions else None
    )
    return {"segments": segments, "speech_end": last_speech_end}


def delete_video(video_file: Path, name: str):
    """Delete the source video after its transcript has been verified."""
    video_file.unlink()
    print(f"[{name}] source video deleted ({video_file.name})")


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


def coverage_warning(
    segments: list, duration: float, speech_end: float | None
) -> str | None:
    """Alert if the transcript stops before the detected speech does (guards
    against truncated output). Leading silence is fine — VAD skips it on
    purpose — so only the timing and tail are checked."""
    if not segments:
        return "no speech was detected at all"
    reference = speech_end if speech_end and speech_end > 0 else duration
    if reference <= 0:
        return None
    last = segments[-1]["end"]
    if last < reference * 0.95:
        return f"last speech ends at {last:.0f}s but detected speech ends at {reference:.0f}s"
    return None


def main(force: bool = False, delete_videos: bool = False):
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
    for video_dir, video_file in jobs:
        name = video_dir.name
        audio_path = video_dir / f"{name}.wav"
        if not audio_path.exists():
            failed += 1
            continue
        duration = durations.get(name, 0.0)
        try:
            t0 = time.time()
            transcript = transcribe(audio_path, duration)
            segments = transcript["segments"]
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
            issue = coverage_warning(segments, duration, transcript.get("speech_end"))
            if issue:
                print(
                    f"[{name}] WARNING: transcript may be incomplete, keeping the "
                    f"source video: {issue}",
                    file=sys.stderr,
                )
            elif delete_videos:
                delete_video(video_file, name)
        except Exception as e:
            failed += 1
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
        finally:
            audio_path.unlink(missing_ok=True)

    print(f"\nAll done! {done} newly transcribed, {failed} failed.")


if __name__ == "__main__":
    main(force="--force" in sys.argv, delete_videos="--delete-videos" in sys.argv)
