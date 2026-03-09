# -*- coding: utf-8 -*-
"""
Configuration and constants for Speaker Identification System.
"""

import os
import threading

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("Warning: python-dotenv not installed, using system environment only")

import torch

# =============================================================================
# Device Detection
# =============================================================================

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
else:
    device = "cpu"

# =============================================================================
# Model Configuration
# =============================================================================

# SpeechBrain ECAPA-TDNN model (trained on VoxCeleb, EER ~0.8%)
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

# =============================================================================
# File Configuration
# =============================================================================

UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

# Persistence file for enrolled speakers
SPEAKER_MEMORY_FILE = 'enrolled_speakers.pkl'

# =============================================================================
# Speaker Detection Thresholds
# =============================================================================

# Session-based speaker clustering
SPEAKER_CLUSTER_THRESHOLD = 0.15  # Very aggressive for unstable embeddings

# Adaptive speaker profile settings
ADAPTIVE_UPDATE_THRESHOLD = 0.60  # Min similarity to trigger update
ADAPTIVE_LEARNING_RATE = 0.1  # How much to weight new embedding (0.1 = 10% new, 90% old)

# Audio accumulation for stable speaker ID
ACCUM_DURATION_FOR_SPEAKER_ID = 2.0  # seconds (was 3.0, reduced for faster speaker change)

# Chunk history settings
MAX_CHUNK_HISTORY = 15  # Keep last 15 chunks (~15 seconds)

# Session speaker limits (prevent memory leak)
MAX_SESSION_SPEAKERS = 50  # Maximum unique speakers per session

# =============================================================================
# Threading
# =============================================================================

processing_lock = threading.Lock()  # Prevent concurrent GPU access

# =============================================================================
# Valid ASR Models
# =============================================================================

# Valid Parakeet models
PARAKEET_MODELS = [
    'mlx-community/parakeet-tdt-1.1b',
    'mlx-community/parakeet-tdt-0.6b-v3',
    'mlx-community/parakeet-ctc-1.1b',
    'mlx-community/parakeet-ctc-0.6b',
    'mlx-community/parakeet-tdt-0.6b-v2'
]

# Valid Voxtral models
VOXTRAL_MODELS = [
    'mlx-community/Voxtral-Mini-3B-2507-bf16',
    'mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit'
]

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
