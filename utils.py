# -*- coding: utf-8 -*-
"""
Shared utility functions for Speaker Identification System.
"""

import io
import re
import hashlib
import numpy as np
from pydub import AudioSegment

from config import ALLOWED_EXTENSIONS


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def embedding_to_hash(embedding, length=8):
    """Convert embedding to short hex hash for human-readable identification."""
    # Handle both numpy arrays and PyTorch tensors
    if hasattr(embedding, 'cpu'):  # PyTorch tensor
        emb_np = embedding.cpu().numpy()
    else:
        emb_np = embedding
    emb_bytes = emb_np.astype(np.float32).tobytes()
    full_hash = hashlib.sha256(emb_bytes).hexdigest()
    return full_hash[:length].upper()


def convert_webm_to_wav(audio_data):
    """Convert WebM audio data to WAV buffer.
    
    Args:
        audio_data: bytes of WebM audio
        
    Returns:
        io.BytesIO buffer containing WAV audio at 16kHz mono
    """
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    
    return wav_buffer


def extract_name_from_text(text):
    """
    Extract a name from spoken text like:
    - "My name is Akash"
    - "I am John"
    - "Call me Sarah"
    - "This is Mike speaking"
    """
    text = text.strip()
    
    # Patterns to match name introductions
    patterns = [
        r"(?:my name is|i'm|i am|this is|call me|it's|its)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:my name is|i'm|i am|this is|call me|it's|its)\s+([a-z]+)",
        r"^([A-Z][a-z]+)\s+(?:here|speaking)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Capitalize first letter of each word
            name = ' '.join(word.capitalize() for word in name.split())
            return name
    
    return None


def extract_youtube_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
