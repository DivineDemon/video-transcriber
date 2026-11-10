#!/usr/bin/env python3
"""
Auto-transcribe all videos in the videos/ directory.
Detects videos without transcripts and transcribes them.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from faster_whisper import WhisperModel


VIDEOS_DIR = Path("/Users/mushood/Documents/code/personal/video-transcriber/videos")
MODEL_SIZE = "large-v3"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ]
    print(f"Extracting audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return False
    print(f"Audio extracted to {audio_path}")
    return True


def transcribe_audio(audio_path: Path) -> list:
    """Transcribe audio using faster-whisper."""
    print(f"Loading model: {MODEL_SIZE}")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    print("Transcribing...")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in segment.words
            ] if segment.words else []
        })
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}")
    
    return results


def format_transcript(segments: list, output_path: Path):
    """Write formatted transcript to file."""
    with open(output_path, "w") as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
    print(f"Transcript saved to {output_path}")


def is_done(video_dir: Path) -> bool:
    """Check if transcription is already complete for this video directory."""
    transcript_txt = video_dir / f"{video_dir.name}-transcript.txt"
    transcript_json = video_dir / f"{video_dir.name}-transcript.json"
    return transcript_txt.exists() and transcript_json.exists()


def find_video_file(video_dir: Path) -> Path | None:
    """Find the video file in the directory."""
    for ext in (".mov", ".mp4", ".mkv", ".avi", ".webm"):
        matches = list(video_dir.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def process_video_dir(video_dir: Path):
    """Process a single video directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {video_dir.name}")
    print(f"{'='*60}")
    
    if is_done(video_dir):
        print("Already transcribed. Skipping.")
        return
    
    video_file = find_video_file(video_dir)
    if not video_file:
        print(f"No video file found in {video_dir}")
        return
    
    audio_path = video_dir / f"{video_dir.name}.wav"
    transcript_txt = video_dir / f"{video_dir.name}-transcript.txt"
    transcript_json = video_dir / f"{video_dir.name}-transcript.json"
    
    # Extract audio
    if not extract_audio(video_file, audio_path):
        return
    
    try:
        # Transcribe
        segments = transcribe_audio(audio_path)
        
        # Save transcript
        format_transcript(segments, transcript_txt)
        
        # Save JSON
        with open(transcript_json, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"JSON transcript saved to {transcript_json}")
        
    finally:
        # Clean up audio file
        if audio_path.exists():
            audio_path.unlink()
            print(f"Cleaned up temporary audio file")


def main():
    if not VIDEOS_DIR.exists():
        print(f"Videos directory not found: {VIDEOS_DIR}")
        sys.exit(1)
    
    # Find all subdirectories in videos/
    video_dirs = [d for d in VIDEOS_DIR.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("No video directories found.")
        return
    
    print(f"Found {len(video_dirs)} video directory(ies):")
    for d in video_dirs:
        status = "DONE" if is_done(d) else "PENDING"
        print(f"  - {d.name} [{status}]")
    
    for video_dir in video_dirs:
        process_video_dir(video_dir)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()