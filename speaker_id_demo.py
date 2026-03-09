# -*- coding: utf-8 -*-
"""
Speaker Identification System

A speaker verification system using Microsoft's WavLM model.
Supports enrolling speakers, identifying speakers from audio,
and streaming speaker tracking with diarization.

Requirements (install via pip):
    pip install torch torchaudio transformers librosa soundfile numpy requests
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import requests
from torch.nn.functional import cosine_similarity
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

# =============================================================================
# Global Model Setup
# =============================================================================

MODEL_ID = "microsoft/wavlm-base-plus-sv"  # speaker verification / x-vector model

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# These will be initialized when init_model() is called
feature_extractor = None
spk_model = None


def init_model():
    """Initialize the speaker embedding model. Call this before using other functions."""
    global feature_extractor, spk_model
    
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_ID}...")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    spk_model = AutoModelForAudioXVector.from_pretrained(MODEL_ID).to(device)
    spk_model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Feature extractor type: {type(feature_extractor)}")


# =============================================================================
# Audio Helper Functions
# =============================================================================

def load_audio_mono(path, target_sr=16000):
    """Load audio file as mono waveform."""
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav, target_sr


@torch.no_grad()
def extract_embedding(wav, sr=16000):
    """
    Extract a fixed-size speaker embedding from audio waveform.
    
    Args:
        wav: 1D numpy array of audio samples
        sr: Sample rate (default 16000)
    
    Returns:
        Normalized embedding tensor of shape (D,)
    """
    if feature_extractor is None or spk_model is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")
    
    inputs = feature_extractor(
        wav,
        sampling_rate=sr,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = spk_model(**inputs)

    # For AudioXVector models, the embedding is typically in `embeddings`
    if hasattr(outputs, "embeddings"):
        emb = outputs.embeddings
    else:
        # Fallback: use last_hidden_state pooled
        emb = outputs.last_hidden_state.mean(dim=1)

    # L2-normalize
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.squeeze(0)  # shape: (D,)


def cosine_sim(e1, e2):
    """Compute cosine similarity between two embeddings."""
    e1 = e1.unsqueeze(0) if e1.ndim == 1 else e1
    e2 = e2.unsqueeze(0) if e2.ndim == 1 else e2
    return float(cosine_similarity(e1, e2).item())


def download_wav(url, filename):
    """Download a WAV file from URL."""
    audio = requests.get(url)
    with open(filename, "wb") as f:
        f.write(audio.content)
    print(f"Downloaded: {filename}")


# =============================================================================
# Speaker Memory Class
# =============================================================================

class SpeakerMemory:
    """
    A simple speaker database for enrolling and identifying speakers.
    
    Supports two modes:
    - Array mode: Pass numpy arrays directly (for streaming)
    - File mode: Pass file paths (for batch processing)
    """
    
    def __init__(self, threshold=0.5):
        """
        Initialize speaker memory.
        
        Args:
            threshold: Minimum similarity score to identify a speaker (default 0.5)
        """
        self.db = {}  # name -> embedding
        self.threshold = threshold

    def _embed_from_array(self, wav, sr=16000):
        """Get embedding from numpy array."""
        return extract_embedding(wav, sr)

    def _embed_from_file(self, path, sr=16000):
        """Get embedding from audio file."""
        wav, sr = librosa.load(path, sr=sr, mono=True)
        return extract_embedding(wav, sr)

    def enroll_from_array(self, name, wav, sr=16000):
        """
        Enroll a speaker from a numpy array.
        
        Args:
            name: Speaker name/identifier
            wav: 1D numpy array of audio samples
            sr: Sample rate
        """
        emb = self._embed_from_array(wav, sr)
        self.db[name] = emb
        print(f"Enrolled speaker '{name}' with embedding dim={emb.shape[0]}")

    def enroll_from_file(self, name, path, sr=16000):
        """
        Enroll a speaker from an audio file.
        
        Args:
            name: Speaker name/identifier
            path: Path to audio file
            sr: Sample rate
        """
        emb = self._embed_from_file(path, sr)
        self.db[name] = emb
        print(f"Enrolled speaker '{name}' with embedding dim={emb.shape[-1]}")

    def identify_from_array(self, wav, sr=16000, threshold=None):
        """
        Identify speaker from numpy array.
        
        Args:
            wav: 1D numpy array of audio samples
            sr: Sample rate
            threshold: Override default threshold (optional)
        
        Returns:
            Tuple of (speaker_name, similarity_score)
        """
        if not self.db:
            return "Unknown", 0.0

        threshold = threshold if threshold is not None else self.threshold
        emb = self._embed_from_array(wav, sr)

        best_name = "Unknown"
        best_score = -1.0

        for name, ref_emb in self.db.items():
            score = cosine_sim(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold:
            return best_name, best_score
        else:
            return "Unknown", best_score

    def identify_from_file(self, path, sr=16000, threshold=None):
        """
        Identify speaker from audio file.
        
        Args:
            path: Path to audio file
            sr: Sample rate
            threshold: Override default threshold (optional)
        
        Returns:
            Tuple of (speaker_name, similarity_score)
        """
        if not self.db:
            return "Unknown", 0.0

        threshold = threshold if threshold is not None else self.threshold
        emb = self._embed_from_file(path, sr)

        best_name = "Unknown"
        best_sim = -1.0

        for name, ref_emb in self.db.items():
            sim = cosine_sim(emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < threshold:
            return "Unknown", best_sim

        return best_name, best_sim

    # Convenience aliases
    def enroll(self, name, wav_or_path, sr=16000):
        """Enroll speaker (auto-detect array vs file path)."""
        if isinstance(wav_or_path, (str, bytes)):
            self.enroll_from_file(name, wav_or_path, sr)
        else:
            self.enroll_from_array(name, wav_or_path, sr)

    def identify(self, wav_or_path, sr=16000, threshold=None):
        """Identify speaker (auto-detect array vs file path)."""
        if isinstance(wav_or_path, (str, bytes)):
            return self.identify_from_file(wav_or_path, sr, threshold)
        else:
            return self.identify_from_array(wav_or_path, sr, threshold)

    def list_speakers(self):
        """List all enrolled speakers."""
        return list(self.db.keys())

    def clear(self):
        """Clear all enrolled speakers."""
        self.db.clear()
        print("Speaker memory cleared.")


# =============================================================================
# Streaming Speaker Tracker
# =============================================================================

class StreamingSpeakerTracker:
    """
    Streaming speaker tracker with smoothing to prevent rapid speaker switching.
    
    Processes audio in chunks and maintains a stable "current speaker" state
    that only changes after consecutive confirmations.
    """
    
    def __init__(
        self,
        speaker_memory,
        sr=16000,
        window_sec=0.5,
        hop_sec=0.25,
        threshold=0.5,
        required_confirm=2,
    ):
        """
        Initialize the streaming tracker.
        
        Args:
            speaker_memory: SpeakerMemory instance with enrolled speakers
            sr: Sample rate
            window_sec: Window size for analysis
            hop_sec: Hop size between windows
            threshold: Similarity threshold for identification
            required_confirm: Number of consecutive windows needed to switch speaker
        """
        self.speaker_memory = speaker_memory
        self.sr = sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.threshold = threshold
        self.required_confirm = required_confirm

        # Smoothing state
        self.current_speaker = "Unknown"
        self._pending_speaker = None
        self._pending_count = 0

        # Time bookkeeping
        self._step_idx = 0
        self._t_cursor = 0.0

    def reset(self):
        """Reset tracker state."""
        self.current_speaker = "Unknown"
        self._pending_speaker = None
        self._pending_count = 0
        self._step_idx = 0
        self._t_cursor = 0.0

    def step(self, chunk):
        """
        Process one audio chunk and update speaker state.
        
        Args:
            chunk: 1D numpy array or torch tensor of audio samples
        
        Returns:
            Tuple of (start_time, end_time, chunk_speaker, similarity, current_speaker)
        """
        # Convert to numpy
        if hasattr(chunk, "detach"):
            chunk = chunk.detach().cpu().numpy()
        elif isinstance(chunk, list):
            chunk = np.array(chunk, dtype=np.float32)

        # Identify speaker for this chunk
        spk_name, score = self.speaker_memory.identify_from_array(chunk, sr=self.sr)

        # Time interval for this chunk
        t0 = self._t_cursor
        t1 = t0 + self.window_sec

        # Smoothing logic
        if spk_name != self.current_speaker:
            if spk_name == self._pending_speaker:
                self._pending_count += 1
            else:
                self._pending_speaker = spk_name
                self._pending_count = 1

            if self._pending_count >= self.required_confirm:
                self.current_speaker = self._pending_speaker
                self._pending_speaker = None
                self._pending_count = 0
        else:
            self._pending_speaker = None
            self._pending_count = 0

        # Update time & step counter
        self._step_idx += 1
        self._t_cursor += self.hop_sec

        return t0, t1, spk_name, score, self.current_speaker


# =============================================================================
# Utility Functions
# =============================================================================

def generate_synthetic_voices(output_dir="."):
    """
    Generate synthetic test voices (sine waves with noise).
    
    Returns:
        Dict of filename -> numpy array
    """
    import os
    
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create two simple tones + noise as fake voices
    yumi_freq = 440.0   # A4 (higher pitch)
    akash_freq = 220.0  # A3 (lower pitch)

    yumi_wave = 0.2 * np.sin(2 * np.pi * yumi_freq * t) + 0.02 * np.random.randn(len(t))
    akash_wave = 0.2 * np.sin(2 * np.pi * akash_freq * t) + 0.02 * np.random.randn(len(t))

    # Save as WAV files
    yumi_path = os.path.join(output_dir, "speaker_yumi.wav")
    akash_path = os.path.join(output_dir, "speaker_akash.wav")
    
    sf.write(yumi_path, yumi_wave, sr)
    sf.write(akash_path, akash_wave, sr)

    print(f"Generated synthetic WAVs:")
    print(f"  - {yumi_path}")
    print(f"  - {akash_path}")

    return {
        "speaker_yumi.wav": yumi_wave,
        "speaker_akash.wav": akash_wave,
    }


def tag_text_with_speaker(
    wav_1d,
    tracker,
    sr=16000,
    window_sec=0.5,
    hop_sec=0.25,
    text_generator=None,
):
    """
    Process audio with speaker tagging (simulated ASR + speaker identification).
    
    Args:
        wav_1d: 1D numpy array of audio
        tracker: StreamingSpeakerTracker instance
        sr: Sample rate
        window_sec: Window size
        hop_sec: Hop size
        text_generator: Optional function(chunk, step) -> str for ASR
    
    Returns:
        List of dicts with step, timestamps, speaker info, and text
    """
    total_len = len(wav_1d)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)

    results = []
    start = 0
    step = 0

    print(f"\nProcessing audio with speaker tags")
    print(f"  Total duration: {total_len / sr:.2f} sec")
    print(f"  Window: {window_sec:.2f}s | Hop: {hop_sec:.2f}s\n")

    while start + win <= total_len:
        chunk = wav_1d[start:start + win]
        t0, t1, chunk_spk, score, current_spk = tracker.step(chunk)

        # Generate text (use provided function or placeholder)
        if text_generator:
            text = text_generator(chunk, step)
        else:
            text = f"Chunk #{step}"

        print(
            f"[{step:02d}] {t0:4.2f}-{t1:4.2f} sec | "
            f"speaker={current_spk:12s} (raw={chunk_spk:12s}, sim={score:.3f}) | text='{text}'"
        )

        results.append({
            "step": step,
            "t0": t0,
            "t1": t1,
            "raw_speaker": chunk_spk,
            "current_speaker": current_spk,
            "similarity": float(score),
            "text": text,
        })

        step += 1
        start += hop

    return results


# =============================================================================
# Demo / Main
# =============================================================================

def run_demo():
    """Run a demonstration of the speaker identification system."""
    
    # Initialize model
    init_model()
    
    # Create speaker memory
    speaker_memory = SpeakerMemory(threshold=0.5)
    
    # Generate and enroll synthetic voices
    print("\n" + "="*50)
    print("Generating synthetic test voices...")
    print("="*50)
    
    voices = generate_synthetic_voices()
    
    for fname, wav in voices.items():
        speaker_name = fname.replace(".wav", "")
        speaker_memory.enroll_from_array(speaker_name, wav, sr=16000)
    
    print(f"\nEnrolled speakers: {speaker_memory.list_speakers()}")
    
    # Test identification
    print("\n" + "="*50)
    print("Testing speaker identification...")
    print("="*50)
    
    for fname, wav in voices.items():
        name, score = speaker_memory.identify_from_array(wav, sr=16000)
        print(f"  {fname}: identified as '{name}' (similarity: {score:.3f})")
    
    # Create mixed conversation
    print("\n" + "="*50)
    print("Testing streaming speaker tracking...")
    print("="*50)
    
    sr = 16000
    half_samples = int(sr * 1.5)
    
    yumi_wav = voices["speaker_yumi.wav"][:half_samples]
    akash_wav = voices["speaker_akash.wav"][:half_samples]
    mixed_conv = np.concatenate([yumi_wav, akash_wav])
    
    print(f"\nMixed clip: {len(mixed_conv)/sr:.2f} sec (yumi -> akash)")
    
    # Create tracker and process
    tracker = StreamingSpeakerTracker(
        speaker_memory,
        sr=sr,
        window_sec=0.5,
        hop_sec=0.25,
        threshold=0.5,
        required_confirm=2,
    )
    
    results = tag_text_with_speaker(
        mixed_conv,
        tracker,
        sr=sr,
        window_sec=0.5,
        hop_sec=0.25,
    )
    
    print("\nDemo completed!")
    return speaker_memory, results


if __name__ == "__main__":
    run_demo()