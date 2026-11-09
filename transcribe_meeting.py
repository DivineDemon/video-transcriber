#!/usr/bin/env python3
"""
Transcribe a video/audio file using faster-whisper.
Extracts audio with ffmpeg, then transcribes with word-level timestamps.
"""

import subprocess
import sys
import os
from pathlib import Path
from faster_whisper import WhisperModel


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    print(f"Extracting audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return False
    print(f"Audio extracted to {audio_path}")
    return True


def transcribe_audio(audio_path: str, model_size: str = "large-v3") -> list:
    """Transcribe audio using faster-whisper."""
    print(f"Loading model: {model_size}")
    # Use CPU with int8 quantization for speed/memory balance
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    print("Transcribing...")
    segments, info = model.transcribe(
        audio_path,
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


def format_transcript(segments: list, output_path: str):
    """Write formatted transcript to file."""
    with open(output_path, "w") as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
    print(f"Transcript saved to {output_path}")


def main():
    video_path = "/Users/mushood/Desktop/karunya-meeting.mov"
    audio_path = "/Users/mushood/Desktop/karunya-meeting.wav"
    transcript_path = "/Users/mushood/Desktop/karunya-meeting-transcript.txt"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)
    
    # Extract audio
    if not extract_audio(video_path, audio_path):
        sys.exit(1)
    
    try:
        # Transcribe
        segments = transcribe_audio(audio_path, model_size="large-v3")
        
        # Save transcript
        format_transcript(segments, transcript_path)
        
        # Also save JSON for programmatic use
        import json
        json_path = "/Users/mushood/Desktop/karunya-meeting-transcript.json"
        with open(json_path, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"JSON transcript saved to {json_path}")
        
    finally:
        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Cleaned up temporary audio file")


if __name__ == "__main__":
    main()