# -*- coding: utf-8 -*-
"""
YouTube audio processing for Speaker Identification System.
"""

import os
import numpy as np
import soundfile as sf

from config import UPLOAD_FOLDER
from audio import load_audio
from models import (
    transcribe_with_parakeet,
    extract_embedding,
    cosine_sim,
    is_model_loaded,
    PARAKEET_AVAILABLE
)
from speakers import get_speaker_memory

# YouTube download
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
    print("yt-dlp available for YouTube processing")
except ImportError:
    YTDLP_AVAILABLE = False
    print("yt-dlp not available (install with: pip install yt-dlp)")

# Global jobs storage
youtube_jobs = {}  # job_id -> job status/results


def download_youtube_audio(url, output_path):
    """Download audio from YouTube video."""
    if not YTDLP_AVAILABLE:
        raise RuntimeError("yt-dlp not installed")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.wav', ''),
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'duration_str': f"{info.get('duration', 0) // 60}:{info.get('duration', 0) % 60:02d}"
        }


def process_youtube_audio(job_id, audio_path, segment_duration=10):
    """Process YouTube audio: segment, transcribe, and identify speakers."""
    global youtube_jobs
    
    try:
        youtube_jobs[job_id]['status'] = 'processing'
        youtube_jobs[job_id]['message'] = 'Loading audio...'
        
        # Load full audio
        import librosa
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(wav) / sr
        
        # Split into segments
        segment_samples = int(segment_duration * sr)
        num_segments = int(np.ceil(len(wav) / segment_samples))
        
        transcript = []
        speaker_counts = {}
        speaker_memory = get_speaker_memory()
        
        for i in range(num_segments):
            if job_id not in youtube_jobs or youtube_jobs[job_id].get('cancelled'):
                break
            
            start_sample = i * segment_samples
            end_sample = min((i + 1) * segment_samples, len(wav))
            segment = wav[start_sample:end_sample]
            
            # Update progress
            progress = (i + 1) / num_segments * 100
            youtube_jobs[job_id]['progress'] = progress
            youtube_jobs[job_id]['message'] = f'Processing segment {i+1}/{num_segments}'
            
            # Skip silent segments
            if np.max(np.abs(segment)) < 0.01:
                continue
            
            # Transcribe segment
            text = None
            if PARAKEET_AVAILABLE:
                # Save temp file for Parakeet
                temp_path = os.path.join(UPLOAD_FOLDER, f'yt_seg_{job_id}_{i}.wav')
                sf.write(temp_path, segment, sr)
                text = transcribe_with_parakeet(temp_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if not text or text.strip() == '':
                continue
            
            # Identify speaker
            speaker = 'Unknown'
            if is_model_loaded() and speaker_memory:
                try:
                    emb = extract_embedding(segment, sr)
                    best_name = 'Unknown'
                    best_score = -1.0
                    threshold = 0.6  # Lower threshold for clean audio
                    
                    for name, ref_emb in speaker_memory.items():
                        score = cosine_sim(emb, ref_emb)
                        if score > best_score:
                            best_score = score
                            best_name = name
                    
                    if best_score >= threshold:
                        speaker = best_name
                except Exception as e:
                    print(f"Speaker ID error: {e}")
            
            # Track speaker counts
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            # Calculate timestamp
            start_time = start_sample / sr
            time_str = f"{int(start_time // 60)}:{int(start_time % 60):02d}"
            
            transcript.append({
                'time': time_str,
                'time_seconds': start_time,
                'speaker': speaker,
                'text': text
            })
            
            # Update partial results
            youtube_jobs[job_id]['transcript'] = transcript
        
        # Build speaker list
        speakers = [{'name': name, 'segments': count} for name, count in speaker_counts.items()]
        speakers.sort(key=lambda x: x['segments'], reverse=True)
        
        youtube_jobs[job_id]['status'] = 'complete'
        youtube_jobs[job_id]['transcript'] = transcript
        youtube_jobs[job_id]['speakers'] = speakers
        youtube_jobs[job_id]['message'] = 'Complete'
        
        # Don't delete audio file - keep for playback
            
    except Exception as e:
        print(f"YouTube processing error: {e}")
        import traceback
        traceback.print_exc()
        youtube_jobs[job_id]['status'] = 'error'
        youtube_jobs[job_id]['error'] = str(e)


def get_youtube_jobs():
    """Get all YouTube jobs."""
    return youtube_jobs


def get_youtube_job(job_id):
    """Get a specific YouTube job."""
    return youtube_jobs.get(job_id)


def set_youtube_job(job_id, job_data):
    """Set a YouTube job."""
    youtube_jobs[job_id] = job_data


def is_ytdlp_available():
    """Check if yt-dlp is available."""
    return YTDLP_AVAILABLE
