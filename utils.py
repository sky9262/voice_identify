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

# Speaker name validation constants
MAX_SPEAKER_NAME_LENGTH = 50
ALLOWED_NAME_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-\'')


def sanitize_speaker_name(name):
    """Sanitize and validate speaker name.
    
    Args:
        name: Raw speaker name input
        
    Returns:
        tuple: (sanitized_name, error_message)
               If valid, error_message is None
               If invalid, sanitized_name is None
    """
    if not name:
        return None, 'Speaker name is required'
    
    # Strip whitespace
    name = name.strip()
    
    if not name:
        return None, 'Speaker name cannot be empty'
    
    # Check length
    if len(name) > MAX_SPEAKER_NAME_LENGTH:
        return None, f'Speaker name too long (max {MAX_SPEAKER_NAME_LENGTH} characters)'
    
    if len(name) < 2:
        return None, 'Speaker name must be at least 2 characters'
    
    # Check for invalid characters
    invalid_chars = set(name) - ALLOWED_NAME_CHARS
    if invalid_chars:
        return None, f'Speaker name contains invalid characters: {", ".join(repr(c) for c in invalid_chars)}'
    
    # Prevent HTML/script injection patterns
    dangerous_patterns = ['<', '>', 'script', 'javascript:', 'onclick', 'onerror', 'onload']
    name_lower = name.lower()
    for pattern in dangerous_patterns:
        if pattern in name_lower:
            return None, f'Speaker name contains disallowed pattern'
    
    # Collapse multiple spaces
    import re
    name = re.sub(r'\s+', ' ', name)
    
    return name, None


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
