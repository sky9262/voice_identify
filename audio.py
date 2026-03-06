# -*- coding: utf-8 -*-
"""
Audio processing functions for Speaker Identification System.
"""

import io
import numpy as np
import librosa

# Noise reduction
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    print("Warning: noisereduce not installed. Run: pip install noisereduce")


def reduce_noise(audio, sr=16000, level=80):
    """Apply noise reduction to audio signal.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        level: Noise reduction level (0-100). 0=off, 100=maximum
    """
    if not NOISE_REDUCE_AVAILABLE or level <= 0:
        return audio
    
    try:
        # Convert level (0-100) to prop_decrease (0-1)
        # level 0 = no reduction, level 100 = maximum reduction
        prop_decrease = level / 100.0
        
        # Adjust n_std_thresh based on level (lower = more aggressive)
        # level 100 -> thresh 0.5 (very aggressive)
        # level 50 -> thresh 1.5 (moderate)
        # level 20 -> thresh 2.5 (light)
        n_std_thresh = 3.0 - (level / 50.0)
        n_std_thresh = max(0.5, min(3.0, n_std_thresh))
        
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=True,
            n_fft=512,
            hop_length=128,
            n_std_thresh_stationary=n_std_thresh,
        )
        return reduced
    except Exception as e:
        print(f"Noise reduction failed: {e}")
        return audio


def load_audio(filepath, target_sr=16000):
    """Load audio file and convert to mono."""
    wav, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return wav, target_sr


def load_audio_with_noise_reduction(filepath_or_buffer, target_sr=16000, apply_nr=True):
    """Load audio and optionally apply noise reduction."""
    wav, sr = librosa.load(filepath_or_buffer, sr=target_sr, mono=True)
    
    if apply_nr:
        wav = reduce_noise(wav, sr)
    
    return wav, sr


def load_audio_from_bytes(audio_data, audio_format='webm', target_sr=16000, noise_level=80):
    """Load audio from bytes (WebM or WAV) with noise reduction.
    
    Args:
        audio_data: bytes of audio
        audio_format: 'webm' or 'wav'
        target_sr: target sample rate
        noise_level: noise reduction level (0-100)
        
    Returns:
        tuple: (wav array, sample rate)
    """
    from pydub import AudioSegment
    
    if audio_format == 'wav':
        # Direct WAV - just load with librosa
        wav, sr = librosa.load(io.BytesIO(audio_data), sr=target_sr, mono=True)
    else:
        # Convert WebM/Opus to WAV using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        audio_segment = audio_segment.set_frame_rate(target_sr).set_channels(1)
        
        # Export to WAV bytes
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Load with librosa
        wav, sr = librosa.load(wav_buffer, sr=target_sr, mono=True)
    
    # Apply noise reduction
    if noise_level > 0:
        wav = reduce_noise(wav, sr, level=noise_level)
    
    return wav, sr


def is_audio_valid(wav, sr, min_samples=1600, min_amplitude=0.01):
    """Check if audio is valid (not too short or silent).
    
    Args:
        wav: audio waveform
        sr: sample rate
        min_samples: minimum number of samples (default 1600 = 0.1s at 16kHz)
        min_amplitude: minimum peak amplitude
        
    Returns:
        tuple: (is_valid, reason)
    """
    if len(wav) < min_samples:
        return False, 'Audio too short'
    
    if np.max(np.abs(wav)) < min_amplitude:
        return False, 'No audio detected'
    
    return True, None
