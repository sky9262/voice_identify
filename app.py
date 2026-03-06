# -*- coding: utf-8 -*-
"""
Speaker Identification Web Server

Using SpeechBrain ECAPA-TDNN for high-accuracy speaker verification.
"""

import os
import io
import re
import json
import uuid
import threading

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("Warning: python-dotenv not installed, using system environment only")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment

# Fix torchaudio compatibility for SpeechBrain
try:
    import torchaudio
    # SpeechBrain may need this attribute
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile']
except ImportError:
    pass

# SpeechBrain ECAPA-TDNN (high-accuracy speaker verification)
SPEECHBRAIN_AVAILABLE = False
try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
    print("SpeechBrain ECAPA-TDNN available")
except ImportError as e:
    print(f"SpeechBrain not available: {e}")
except Exception as e:
    print(f"SpeechBrain error: {e}")

# Optional: OpenAI for better transcription
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Noise reduction
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False

# YouTube download
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
    print("yt-dlp available for YouTube processing")
except ImportError:
    YTDLP_AVAILABLE = False
    print("yt-dlp not available (install with: pip install yt-dlp)")
    print("Warning: noisereduce not installed. Run: pip install noisereduce")

# Parakeet-MLX (Apple Silicon optimized)
PARAKEET_AVAILABLE = False
parakeet_model = None
current_parakeet_model_name = "mlx-community/parakeet-tdt-1.1b"  # Track current model
try:
    from parakeet_mlx import from_pretrained as parakeet_from_pretrained
    PARAKEET_AVAILABLE = True
    print("Parakeet-MLX loaded (Apple Silicon GPU optimized)")
except ImportError as e:
    print(f"Warning: parakeet-mlx import failed: {e}")
except Exception as e:
    print(f"Warning: parakeet-mlx error: {e}")

# Voxtral-MLX (Mistral ASR via mlx-audio)
VOXTRAL_AVAILABLE = False
voxtral_model = None
current_voxtral_model_name = None
try:
    from mlx_audio.stt.utils import load_model as voxtral_load_model
    from mlx_audio.stt.generate import generate_transcription as voxtral_generate
    VOXTRAL_AVAILABLE = True
    print("Voxtral-MLX (mlx-audio) available")
except ImportError as e:
    print(f"Warning: mlx-audio import failed: {e}")
except Exception as e:
    print(f"Warning: mlx-audio error: {e}")

# Pyannote Speaker Segmentation (for accurate speaker change detection)
PYANNOTE_AVAILABLE = False
pyannote_segmentation = None
try:
    from pyannote.audio import Model as PyannoteModel
    from pyannote.audio.pipelines.utils import get_model
    PYANNOTE_AVAILABLE = True
    print("Pyannote segmentation available")
except ImportError as e:
    print(f"Warning: pyannote.audio import failed: {e}")
except Exception as e:
    print(f"Warning: pyannote.audio error: {e}")

# =============================================================================
# Flask App Setup
# =============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

# =============================================================================
# Global Model & Speaker Memory
# =============================================================================

# SpeechBrain ECAPA-TDNN model (trained on VoxCeleb, EER ~0.8%)
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

# Detect best available device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
else:
    device = "cpu"

# Global variables
ecapa_model = None  # SpeechBrain ECAPA-TDNN
speaker_memory = {}  # name -> embedding
model_loaded = False
youtube_jobs = {}  # job_id -> job status/results
processing_lock = threading.Lock()  # Prevent concurrent GPU access

# Persistence file for enrolled speakers
SPEAKER_MEMORY_FILE = 'enrolled_speakers.pkl'

# Audio accumulation for stable speaker ID (2+ seconds - reduced for faster detection)
accumulated_audio = []  # List of audio chunks
ACCUM_DURATION_FOR_SPEAKER_ID = 2.0  # seconds (was 3.0, reduced for faster speaker change)

# Chunk history for retroactive split detection (hybrid approach)
chunk_history = []  # List of {embedding, text, index}
MAX_CHUNK_HISTORY = 15  # Keep last 15 chunks (~15 seconds)
last_stable_speaker = None  # Last speaker from 3-second stable detection
chunk_index_counter = 0  # Global counter for chunk ordering

def save_speaker_memory():
    """Save enrolled speakers to disk."""
    import pickle
    try:
        # Convert tensors to numpy for serialization
        save_dict = {}
        for name, emb in speaker_memory.items():
            if hasattr(emb, 'cpu'):
                save_dict[name] = emb.cpu().numpy()
            else:
                save_dict[name] = np.array(emb)
        with open(SPEAKER_MEMORY_FILE, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Saved {len(save_dict)} enrolled speakers to {SPEAKER_MEMORY_FILE}")
    except Exception as e:
        print(f"Error saving speaker memory: {e}")

def load_speaker_memory():
    """Load enrolled speakers from disk."""
    import pickle
    global speaker_memory
    try:
        if os.path.exists(SPEAKER_MEMORY_FILE):
            with open(SPEAKER_MEMORY_FILE, 'rb') as f:
                speaker_memory = pickle.load(f)
            print(f"Loaded {len(speaker_memory)} enrolled speakers from {SPEAKER_MEMORY_FILE}")
    except Exception as e:
        print(f"Error loading speaker memory: {e}")

# Load saved speakers on startup
load_speaker_memory()

# Adaptive speaker profile settings
ADAPTIVE_UPDATE_THRESHOLD = 0.60  # Min similarity to trigger update
ADAPTIVE_LEARNING_RATE = 0.1  # How much to weight new embedding (0.1 = 10% new, 90% old)
adaptive_update_counter = {}  # Track updates per speaker

def update_speaker_profile(speaker_name, new_embedding):
    """Update enrolled speaker embedding with new sample using exponential moving average.
    
    This allows speaker profiles to adapt and improve over time.
    """
    global speaker_memory, adaptive_update_counter
    
    if speaker_name not in speaker_memory:
        return False
    
    # Convert to numpy
    if hasattr(new_embedding, 'cpu'):
        new_emb = new_embedding.cpu().numpy().flatten()
    else:
        new_emb = np.array(new_embedding).flatten()
    
    old_emb = speaker_memory[speaker_name]
    if hasattr(old_emb, 'cpu'):
        old_emb = old_emb.cpu().numpy().flatten()
    else:
        old_emb = np.array(old_emb).flatten()
    
    # Exponential moving average update
    updated_emb = (1 - ADAPTIVE_LEARNING_RATE) * old_emb + ADAPTIVE_LEARNING_RATE * new_emb
    
    # Normalize the embedding
    updated_emb = updated_emb / (np.linalg.norm(updated_emb) + 1e-8)
    
    # Store back
    speaker_memory[speaker_name] = updated_emb
    
    # Track update count
    adaptive_update_counter[speaker_name] = adaptive_update_counter.get(speaker_name, 0) + 1
    count = adaptive_update_counter[speaker_name]
    
    # Save to disk every 10 updates
    if count % 10 == 0:
        save_speaker_memory()
        print(f"Adaptive update: '{speaker_name}' profile saved (update #{count})")
    else:
        print(f"Adaptive update: '{speaker_name}' profile updated (update #{count})")
    
    return True

# Session-based speaker clustering for stable IDs
session_speakers = {}  # hash -> averaged_embedding
session_speaker_counts = {}  # hash -> count (for averaging)
SPEAKER_CLUSTER_THRESHOLD = 0.15  # Very aggressive for unstable embeddings


def embedding_to_hash(embedding, length=8):
    """Convert embedding to short hex hash for human-readable identification."""
    import hashlib
    # Handle both numpy arrays and PyTorch tensors
    if hasattr(embedding, 'cpu'):  # PyTorch tensor
        emb_np = embedding.cpu().numpy()
    else:
        emb_np = embedding
    emb_bytes = emb_np.astype(np.float32).tobytes()
    full_hash = hashlib.sha256(emb_bytes).hexdigest()
    return full_hash[:length].upper()


def get_stable_speaker_id(embedding):
    """
    Find or create a stable speaker ID by clustering similar embeddings.
    Returns a consistent hash for the same speaker across chunks.
    """
    global session_speakers, session_speaker_counts
    
    # Convert to numpy
    if hasattr(embedding, 'cpu'):
        emb_np = embedding.cpu().numpy().flatten()
    else:
        emb_np = np.array(embedding).flatten()
    
    # Compare with existing session speakers
    best_match = None
    best_sim = 0
    
    for speaker_hash, stored_emb in session_speakers.items():
        # Cosine similarity
        sim = float(np.dot(emb_np, stored_emb.flatten()) / 
                   (np.linalg.norm(emb_np) * np.linalg.norm(stored_emb) + 1e-8))
        if sim > best_sim:
            best_sim = sim
            best_match = speaker_hash
    
    if best_match and best_sim >= SPEAKER_CLUSTER_THRESHOLD:
        # Update running average for this speaker
        count = session_speaker_counts[best_match]
        session_speakers[best_match] = (session_speakers[best_match] * count + emb_np) / (count + 1)
        session_speaker_counts[best_match] = count + 1
        print(f"Matched existing speaker {best_match} (sim={best_sim:.3f})")
        return best_match
    else:
        # New speaker - create new hash
        new_hash = embedding_to_hash(emb_np, length=6)
        session_speakers[new_hash] = emb_np.copy()
        session_speaker_counts[new_hash] = 1
        print(f"New speaker detected: {new_hash} (best_sim={best_sim:.3f})")
        return new_hash


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


def load_audio_with_noise_reduction(filepath_or_buffer, target_sr=16000, apply_nr=True):
    """Load audio and optionally apply noise reduction."""
    wav, sr = librosa.load(filepath_or_buffer, sr=target_sr, mono=True)
    
    if apply_nr:
        wav = reduce_noise(wav, sr)
    
    return wav, sr


def init_model():
    """Initialize SpeechBrain ECAPA-TDNN model."""
    global ecapa_model, model_loaded
    
    if model_loaded:
        return True
    
    if not SPEECHBRAIN_AVAILABLE:
        print("ERROR: SpeechBrain not installed!")
        print("Install with: pip install speechbrain")
        return False
    
    print(f"Loading ECAPA-TDNN model...")
    print(f"Model: {ECAPA_MODEL_ID}")
    print(f"Target device: {device}")
    
    try:
        # SpeechBrain uses run_opts for device placement
        # Note: ECAPA-TDNN may have issues with MPS, use CPU as fallback
        if device == "mps":
            # MPS can have compatibility issues, try it first
            try:
                ecapa_model = EncoderClassifier.from_hparams(
                    source=ECAPA_MODEL_ID,
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "mps"}
                )
                print("ECAPA-TDNN loaded on MPS (Apple Silicon GPU)")
            except Exception as mps_error:
                print(f"MPS failed ({mps_error}), falling back to CPU")
                ecapa_model = EncoderClassifier.from_hparams(
                    source=ECAPA_MODEL_ID,
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"}
                )
                print("ECAPA-TDNN loaded on CPU")
        else:
            ecapa_model = EncoderClassifier.from_hparams(
                source=ECAPA_MODEL_ID,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
            print(f"ECAPA-TDNN loaded on {device}")
        
        model_loaded = True
        print("ECAPA-TDNN speaker verification model loaded successfully!")
        print("This model achieves ~0.8% EER on VoxCeleb test set.")
        return True
        
    except Exception as e:
        print(f"Failed to load ECAPA-TDNN: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_pyannote_segmentation(hf_token=None):
    """Initialize Pyannote segmentation model for speaker change detection.
    
    Requires HuggingFace token for gated model access.
    Set HF_TOKEN environment variable or pass token directly.
    """
    global pyannote_segmentation, PYANNOTE_AVAILABLE
    
    if not PYANNOTE_AVAILABLE:
        print("Pyannote not available")
        return False
    
    if pyannote_segmentation is not None:
        return True
    
    # Get token from environment or parameter
    token = hf_token or os.environ.get('huggingface_token') or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    
    if not token:
        print("WARNING: No HuggingFace token found for Pyannote segmentation.")
        print("Set huggingface_token in .env file or HF_TOKEN environment variable")
        return False
    
    try:
        print("Loading Pyannote segmentation-3.0 model...")
        print(f"Using HuggingFace token: {token[:10]}...")
        pyannote_segmentation = PyannoteModel.from_pretrained(
            "pyannote/segmentation-3.0",
            token=token  # Use 'token' not 'use_auth_token'
        )
        # Move to appropriate device
        if device == "mps":
            # Pyannote may have MPS issues, use CPU
            pyannote_segmentation = pyannote_segmentation.to("cpu")
            print("Pyannote segmentation loaded on CPU (MPS fallback)")
        else:
            pyannote_segmentation = pyannote_segmentation.to(device)
            print(f"Pyannote segmentation loaded on {device}")
        
        print("Pyannote segmentation-3.0 loaded successfully!")
        print("This enables accurate speaker change detection.")
        return True
        
    except Exception as e:
        print(f"Failed to load Pyannote segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_parakeet():
    """Initialize Parakeet-MLX model."""
    global parakeet_model, PARAKEET_AVAILABLE, current_parakeet_model_name
    
    if not PARAKEET_AVAILABLE:
        return False
    
    if parakeet_model is not None:
        return True
    
    try:
        print(f"Loading Parakeet-MLX model ({current_parakeet_model_name})...")
        parakeet_model = parakeet_from_pretrained(current_parakeet_model_name)
        print(f"Parakeet-MLX model {current_parakeet_model_name} loaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to load Parakeet-MLX model: {e}")
        import traceback
        traceback.print_exc()
        PARAKEET_AVAILABLE = False
        return False


def switch_parakeet_model(model_name):
    """Switch to a different Parakeet model."""
    global parakeet_model, PARAKEET_AVAILABLE, current_parakeet_model_name, voxtral_model, current_voxtral_model_name
    
    print(f"switch_parakeet_model called with: {model_name}")  # Debug
    print(f"Current model before switch: {current_parakeet_model_name}")  # Debug
    
    if not PARAKEET_AVAILABLE:
        return False, "Parakeet-MLX not available"
    
    try:
        print(f"Switching to Parakeet model: {model_name}...")
        # Unload current models
        parakeet_model = None
        voxtral_model = None
        current_voxtral_model_name = None
        
        # Load new model
        parakeet_model = parakeet_from_pretrained(model_name)
        current_parakeet_model_name = model_name
        print(f"Parakeet model switched to {model_name} successfully!")
        print(f"Current model after switch: {current_parakeet_model_name}")  # Debug
        return True, None
    except Exception as e:
        print(f"Failed to switch Parakeet model: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def switch_voxtral_model(model_name):
    """Switch to a Voxtral model."""
    global voxtral_model, current_voxtral_model_name, parakeet_model, current_parakeet_model_name
    
    if not VOXTRAL_AVAILABLE:
        return False, "Voxtral (mlx-audio) not available. Install with: pip install mlx-audio"
    
    try:
        print(f"Switching to Voxtral model: {model_name}...")
        # Unload other models
        parakeet_model = None
        voxtral_model = None
        
        # Load Voxtral model
        voxtral_model = voxtral_load_model(model_name)
        current_voxtral_model_name = model_name
        current_parakeet_model_name = None
        print(f"Voxtral model switched to {model_name} successfully!")
        return True, None
    except Exception as e:
        print(f"Failed to switch Voxtral model: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def transcribe_with_parakeet(audio_path):
    """Transcribe audio using Parakeet-MLX (Apple Silicon optimized).
    
    Includes confidence filtering to prevent phantom transcriptions (hallucinations).
    """
    global parakeet_model
    
    if not PARAKEET_AVAILABLE:
        print("Parakeet not available")
        return None
    
    if parakeet_model is None:
        print("Parakeet model not loaded - please select and initialize a model in Settings")
        return None
    
    # Minimum confidence threshold to filter hallucinations
    MIN_CONFIDENCE = 0.80
    
    try:
        # Use parakeet-mlx transcribe
        print(f"Transcribing: {audio_path}")
        result = parakeet_model.transcribe(audio_path)
        print(f"Raw transcription result: {result}, type: {type(result)}")
        
        # Handle different return formats
        if result is None:
            return None
        
        # Check for AlignedResult with confidence filtering
        if hasattr(result, 'sentences') and hasattr(result, 'text'):
            text = result.text.strip() if result.text else ''
            if not text:
                return None
            
            # Filter out <unk> tokens (model couldn't understand audio)
            if '<unk>' in text:
                # Remove all <unk> tokens
                text = text.replace('<unk>', '').strip()
                # Collapse multiple spaces
                import re
                text = re.sub(r'\s+', ' ', text).strip()
                print(f"  Filtered <unk> tokens, remaining: '{text}'")
                if not text:
                    return None
            
            # Calculate average confidence from sentences/tokens
            total_confidence = 0
            token_count = 0
            for sentence in result.sentences:
                if hasattr(sentence, 'tokens'):
                    for token in sentence.tokens:
                        if hasattr(token, 'confidence'):
                            total_confidence += token.confidence
                            token_count += 1
            
            if token_count > 0:
                avg_confidence = total_confidence / token_count
                print(f"  Confidence: {avg_confidence:.3f} ({token_count} tokens)")
                
                # Filter low-confidence transcriptions (likely hallucinations)
                if avg_confidence < MIN_CONFIDENCE:
                    print(f"  FILTERED: confidence {avg_confidence:.3f} < {MIN_CONFIDENCE}")
                    return None
                
                # Also filter very short single-word transcriptions that are likely noise
                word_count = len(text.split())
                if word_count == 1 and len(text) <= 4 and avg_confidence < 0.90:
                    print(f"  FILTERED: short word '{text}' with confidence {avg_confidence:.3f}")
                    return None
            
            return text
        
        elif isinstance(result, str):
            return result.strip() if result.strip() else None
        elif isinstance(result, dict) and 'text' in result:
            return result['text'].strip() if result['text'].strip() else None
        elif hasattr(result, 'text'):
            return result.text.strip() if result.text.strip() else None
        elif isinstance(result, list) and len(result) > 0:
            # May return list of segments
            texts = []
            for seg in result:
                if hasattr(seg, 'text'):
                    texts.append(seg.text)
                elif isinstance(seg, dict) and 'text' in seg:
                    texts.append(seg['text'])
                elif isinstance(seg, str):
                    texts.append(seg)
            return ' '.join(texts).strip() if texts else None
        else:
            text = str(result).strip()
            return text if text else None
            
    except Exception as e:
        import traceback
        print(f"Transcription error: {e}")
        traceback.print_exc()
        return None


def transcribe_with_voxtral(audio_path):
    """Transcribe audio using Voxtral (Mistral ASR via mlx-audio)."""
    global voxtral_model
    
    if not VOXTRAL_AVAILABLE:
        print("Voxtral not available")
        return None
    
    if voxtral_model is None:
        print("Voxtral model not loaded - please select and initialize a model in Settings")
        return None
    
    try:
        print(f"Transcribing with Voxtral: {audio_path}")
        
        # Use mlx-audio generate_transcription
        result = voxtral_generate(
            voxtral_model,    # model
            audio_path,       # audio
            "transcript",     # output_path
            "txt",            # format
            False             # verbose
        )
        
        print(f"Voxtral result: {result}")
        
        # Extract text from result
        if result is None:
            return None
        elif hasattr(result, 'text'):
            return result.text.strip() if result.text else None
        elif hasattr(result, 'segments') and result.segments:
            texts = [seg.get('text', '') for seg in result.segments]
            return ' '.join(texts).strip() if texts else None
        elif isinstance(result, str):
            return result.strip() if result.strip() else None
        else:
            return str(result).strip() if str(result).strip() else None
            
    except Exception as e:
        import traceback
        print(f"Voxtral transcription error: {e}")
        traceback.print_exc()
        return None


def transcribe_audio(audio_path):
    """Unified transcription - uses whichever model is loaded."""
    # Try Parakeet first (fastest)
    if parakeet_model is not None:
        return transcribe_with_parakeet(audio_path)
    # Try Voxtral
    if voxtral_model is not None:
        return transcribe_with_voxtral(audio_path)
    return None


@torch.no_grad()
def extract_embedding(wav, sr=16000):
    """Extract speaker embedding using SpeechBrain ECAPA-TDNN."""
    if not model_loaded:
        raise RuntimeError("Model not initialized")
    
    # SpeechBrain expects torch tensor
    if isinstance(wav, np.ndarray):
        wav_tensor = torch.tensor(wav).float()
    else:
        wav_tensor = wav.float()
    
    # Add batch dimension if needed
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    
    # Get embedding from ECAPA-TDNN
    embedding = ecapa_model.encode_batch(wav_tensor)
    
    # Normalize embedding
    embedding = F.normalize(embedding, p=2, dim=-1)
    
    return embedding.squeeze().cpu()


def cosine_sim(e1, e2):
    """Compute cosine similarity between two embeddings."""
    e1 = e1.unsqueeze(0) if e1.ndim == 1 else e1
    e2 = e2.unsqueeze(0) if e2.ndim == 1 else e2
    return float(torch.nn.functional.cosine_similarity(e1, e2).item())


@torch.no_grad()
def detect_speaker_change_pyannote(wav, sr=16000):
    """Detect speaker change probability using Pyannote segmentation.
    
    Returns:
        dict with:
            - has_change: bool, whether speaker change detected
            - change_prob: float, max change probability (0-1)
            - change_frames: list of frame indices with high change probability
    """
    if pyannote_segmentation is None:
        return {'has_change': False, 'change_prob': 0.0, 'change_frames': [], 'available': False}
    
    try:
        # Ensure audio is torch tensor
        if isinstance(wav, np.ndarray):
            wav_tensor = torch.tensor(wav).float()
        else:
            wav_tensor = wav.float()
        
        # Add batch and channel dimension: (batch, channel, time)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)
        elif wav_tensor.dim() == 2:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        # Move to CPU for Pyannote (safer)
        wav_tensor = wav_tensor.to("cpu")
        
        # Get segmentation output
        # Output shape: (batch, num_frames, num_classes)
        # Classes: [speech, non-speech, speaker_change] or similar
        output = pyannote_segmentation(wav_tensor)
        
        # Get speaker change probabilities
        # Pyannote segmentation-3.0 outputs frame-level speaker probabilities
        # We detect change by looking at probability transitions
        probs = output.squeeze().cpu().numpy()  # (num_frames, num_classes)
        
        # For segmentation model, detect change by looking at speaker probability shifts
        if len(probs.shape) == 2:
            # Multi-speaker probabilities - detect when dominant speaker changes
            num_frames, num_speakers = probs.shape
            
            # Find dominant speaker per frame
            dominant = np.argmax(probs, axis=1)
            
            # Detect changes in dominant speaker
            changes = np.where(np.diff(dominant) != 0)[0]
            
            if len(changes) > 0:
                # Get confidence of change (difference in speaker probs)
                change_confidences = []
                for change_frame in changes:
                    if change_frame + 1 < num_frames:
                        # Difference in max probability before/after
                        conf = abs(probs[change_frame].max() - probs[change_frame + 1].max())
                        change_confidences.append(conf)
                
                max_prob = max(change_confidences) if change_confidences else 0.0
                return {
                    'has_change': True,
                    'change_prob': float(max_prob),
                    'change_frames': changes.tolist(),
                    'available': True
                }
        
        return {'has_change': False, 'change_prob': 0.0, 'change_frames': [], 'available': True}
        
    except Exception as e:
        print(f"Pyannote speaker change detection error: {e}")
        import traceback
        traceback.print_exc()
        return {'has_change': False, 'change_prob': 0.0, 'change_frames': [], 'available': False, 'error': str(e)}


def hybrid_speaker_change_detection(wav, sr, current_embedding, prev_embedding, ecapa_threshold=0.5):
    """Hybrid speaker change detection using Pyannote + ECAPA-TDNN.
    
    Combines:
    1. Pyannote segmentation for frame-level change detection
    2. ECAPA-TDNN cosine similarity for speaker verification
    
    Returns:
        dict with:
            - speaker_changed: bool
            - confidence: float (0-1)
            - method: str ('pyannote', 'ecapa', 'both', 'none')
            - ecapa_sim: float, cosine similarity
            - pyannote_prob: float, change probability
    """
    result = {
        'speaker_changed': False,
        'confidence': 0.0,
        'method': 'none',
        'ecapa_sim': 1.0,
        'pyannote_prob': 0.0
    }
    
    # 1. ECAPA-TDNN similarity check
    ecapa_changed = False
    ecapa_sim = 1.0
    if prev_embedding is not None and current_embedding is not None:
        # Handle numpy arrays
        if isinstance(current_embedding, np.ndarray):
            current_emb = torch.tensor(current_embedding).float()
        else:
            current_emb = current_embedding
        if isinstance(prev_embedding, np.ndarray):
            prev_emb = torch.tensor(prev_embedding).float()
        else:
            prev_emb = prev_embedding
        
        ecapa_sim = cosine_sim(current_emb, prev_emb)
        result['ecapa_sim'] = ecapa_sim
        ecapa_changed = ecapa_sim < ecapa_threshold
    
    # 2. Pyannote change detection
    pyannote_result = detect_speaker_change_pyannote(wav, sr)
    pyannote_changed = pyannote_result.get('has_change', False)
    pyannote_prob = pyannote_result.get('change_prob', 0.0)
    result['pyannote_prob'] = pyannote_prob
    pyannote_available = pyannote_result.get('available', False)
    
    # 3. Hybrid decision
    if pyannote_available:
        # Both methods available - require agreement or high confidence
        if pyannote_changed and ecapa_changed:
            # Both agree - high confidence change
            result['speaker_changed'] = True
            result['confidence'] = (1 - ecapa_sim + pyannote_prob) / 2
            result['method'] = 'both'
        elif pyannote_changed and pyannote_prob > 0.7:
            # Strong Pyannote signal alone
            result['speaker_changed'] = True
            result['confidence'] = pyannote_prob
            result['method'] = 'pyannote'
        elif ecapa_changed and ecapa_sim < 0.3:
            # Strong ECAPA signal alone (very low similarity)
            result['speaker_changed'] = True
            result['confidence'] = 1 - ecapa_sim
            result['method'] = 'ecapa'
        else:
            # No strong signal
            result['speaker_changed'] = False
            result['confidence'] = max(1 - ecapa_sim, pyannote_prob) if ecapa_changed or pyannote_changed else 0.0
    else:
        # Pyannote not available - fall back to ECAPA only
        if ecapa_changed:
            result['speaker_changed'] = True
            result['confidence'] = 1 - ecapa_sim
            result['method'] = 'ecapa'
    
    return result


def load_audio(filepath, target_sr=16000):
    """Load audio file and convert to mono."""
    wav, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return wav, target_sr


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/switch-parakeet-model', methods=['POST'])
def api_switch_parakeet_model():
    """Switch to a different ASR model (Parakeet or Voxtral)."""
    data = request.get_json()
    model_name = data.get('model') if data else None
    print(f"API received model switch request: {model_name}")  # Debug
    
    if not model_name:
        return jsonify({'success': False, 'error': 'No model specified'}), 400
    
    # Valid Parakeet models
    parakeet_models = [
        'mlx-community/parakeet-tdt-1.1b',
        'mlx-community/parakeet-tdt-0.6b-v3',
        'mlx-community/parakeet-ctc-1.1b',
        'mlx-community/parakeet-ctc-0.6b',
        'mlx-community/parakeet-tdt-0.6b-v2'
    ]
    
    # Valid Voxtral models
    voxtral_models = [
        'mlx-community/Voxtral-Mini-3B-2507-bf16',
        'mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit'
    ]
    
    all_valid_models = parakeet_models + voxtral_models
    
    if model_name not in all_valid_models:
        return jsonify({'success': False, 'error': f'Invalid model. Valid options: {all_valid_models}'}), 400
    
    # Route to appropriate model loader
    if model_name in parakeet_models:
        success, error = switch_parakeet_model(model_name)
    else:
        success, error = switch_voxtral_model(model_name)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'Switched to {model_name}',
            'model': model_name
        })
    else:
        return jsonify({'success': False, 'error': error}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    # Build speaker list with hashes
    speakers_with_hash = []
    for name, emb in speaker_memory.items():
        speakers_with_hash.append({
            'name': name,
            'hash': embedding_to_hash(emb)
        })
    
    # Determine active model name
    active_model = None
    if parakeet_model is not None:
        active_model = current_parakeet_model_name
    elif voxtral_model is not None:
        active_model = current_voxtral_model_name
    
    return jsonify({
        'model_loaded': model_loaded,
        'speaker_model': 'ECAPA-TDNN' if model_loaded else 'Not loaded',
        'parakeet_available': PARAKEET_AVAILABLE,
        'parakeet_loaded': parakeet_model is not None,
        'parakeet_model_name': current_parakeet_model_name if parakeet_model is not None else None,
        'voxtral_available': VOXTRAL_AVAILABLE,
        'voxtral_loaded': voxtral_model is not None,
        'voxtral_model_name': current_voxtral_model_name if voxtral_model is not None else None,
        'asr_model_loaded': parakeet_model is not None or voxtral_model is not None,
        'asr_model_name': active_model,
        'pyannote_available': PYANNOTE_AVAILABLE,
        'pyannote_loaded': pyannote_segmentation is not None,
        'device': device,
        'speakers_enrolled': speakers_with_hash,
        'speaker_count': len(speaker_memory)
    })


@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize the model."""
    try:
        init_model()
        # Don't auto-init Parakeet - let user choose model in Settings
        # if PARAKEET_AVAILABLE:
        #     init_parakeet()
        return jsonify({'success': True, 'message': 'Model initialized successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/init-pyannote', methods=['POST'])
def initialize_pyannote():
    """Initialize Pyannote segmentation model for hybrid speaker change detection.
    
    Requires HuggingFace token in request body or HF_TOKEN environment variable.
    """
    if not PYANNOTE_AVAILABLE:
        return jsonify({
            'success': False, 
            'error': 'Pyannote not installed. Run: pip install pyannote.audio'
        }), 400
    
    # Get token from request or environment
    data = request.get_json() or {}
    hf_token = data.get('hf_token') or data.get('token')
    
    try:
        success = init_pyannote_segmentation(hf_token)
        if success:
            return jsonify({
                'success': True,
                'message': 'Pyannote segmentation model loaded successfully',
                'model': 'pyannote/segmentation-3.0'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load Pyannote. Check HuggingFace token and model access.',
                'instructions': [
                    '1. Get token at: https://huggingface.co/settings/tokens',
                    '2. Accept terms at: https://huggingface.co/pyannote/segmentation-3.0',
                    '3. Pass token in request body or set HF_TOKEN environment variable'
                ]
            }), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/enroll', methods=['POST'])
def enroll_speaker():
    """Enroll a new speaker."""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    name = request.form.get('name', '').strip()
    
    if not name:
        return jsonify({'success': False, 'error': 'Speaker name is required'}), 400
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        audio_data = file.read()
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Handle WebM from browser recording
        if ext == 'webm':
            # Convert WebM to WAV using pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        else:
            # Save and load other formats directly
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            wav, sr = librosa.load(filepath, sr=16000, mono=True)
            os.remove(filepath)
        
        # Apply noise reduction
        wav = reduce_noise(wav, sr)
        
        # Check audio quality
        duration = len(wav) / sr
        if duration < 1.0:
            return jsonify({'success': False, 'error': f'Recording too short ({duration:.1f}s). Please record at least 2 seconds.'}), 400
        
        if np.max(np.abs(wav)) < 0.01:
            return jsonify({'success': False, 'error': 'No audio detected. Please speak louder.'}), 400
        
        # Extract and store embedding
        emb = extract_embedding(wav, sr)
        speaker_memory[name] = emb
        
        # Save to disk for persistence
        save_speaker_memory()
        
        return jsonify({
            'success': True,
            'message': f"Speaker '{name}' enrolled successfully ({duration:.1f}s audio)",
            'speaker_count': len(speaker_memory)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/enroll-from-hash', methods=['POST'])
def enroll_from_hash():
    """Enroll a speaker using their session hash (fingerprint).
    
    This allows enrolling unenrolled speakers from the transcript by clicking
    on their hash and assigning a name.
    """
    global speaker_memory, session_speakers, adaptive_update_counter
    
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    data = request.get_json()
    speaker_hash = data.get('hash', '').strip().upper()
    name = data.get('name', '').strip()
    merge_with_existing = data.get('merge', False)
    
    if not speaker_hash:
        return jsonify({'success': False, 'error': 'Speaker hash is required'}), 400
    
    if not name:
        return jsonify({'success': False, 'error': 'Speaker name is required'}), 400
    
    # Remove # prefix if present
    if speaker_hash.startswith('#'):
        speaker_hash = speaker_hash[1:]
    
    # Look up embedding from session speakers
    if speaker_hash not in session_speakers:
        return jsonify({'success': False, 'error': f'Unknown speaker hash: {speaker_hash}. Speaker may have expired from session.'}), 400
    
    try:
        session_embedding = session_speakers[speaker_hash]
        
        # Convert to tensor for storage
        if isinstance(session_embedding, np.ndarray):
            emb_tensor = torch.tensor(session_embedding).float()
        else:
            emb_tensor = session_embedding
        
        if merge_with_existing and name in speaker_memory:
            # Use adaptive learning to update existing speaker's embedding (EMA)
            existing_emb = speaker_memory[name]
            if hasattr(existing_emb, 'cpu'):
                existing_np = existing_emb.cpu().numpy().flatten()
            else:
                existing_np = np.array(existing_emb).flatten()
            
            new_emb = session_embedding.flatten()
            
            # Exponential moving average - learn from the new fingerprint
            # Use higher learning rate (0.3) for manual merges since user confirmed identity
            merge_learning_rate = 0.3  # Higher than auto-update (0.1) since user confirmed
            merged = (1 - merge_learning_rate) * existing_np + merge_learning_rate * new_emb
            
            # Normalize the merged embedding
            merged = merged / (np.linalg.norm(merged) + 1e-8)
            speaker_memory[name] = torch.tensor(merged).float()
            
            # Track the update
            adaptive_update_counter[name] = adaptive_update_counter.get(name, 0) + 1
            update_count = adaptive_update_counter[name]
            
            message = f"Learned fingerprint into '{name}' (profile update #{update_count})"
            print(f"Adaptive merge: '{name}' learned from hash #{speaker_hash}, update #{update_count}")
        else:
            # Create new speaker or overwrite
            speaker_memory[name] = emb_tensor
            message = f"Speaker '{name}' enrolled from fingerprint"
        
        # Save to disk
        save_speaker_memory()
        
        return jsonify({
            'success': True,
            'message': message,
            'speaker_count': len(speaker_memory),
            'hash': speaker_hash,
            'name': name
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/session-speakers', methods=['GET'])
def get_session_speakers():
    """Get list of unenrolled session speakers (hashes)."""
    unenrolled = []
    enrolled_hashes = set()
    
    # Get hashes of enrolled speakers
    for name, emb in speaker_memory.items():
        enrolled_hashes.add(embedding_to_hash(emb))
    
    # Get unenrolled session speakers
    for speaker_hash in session_speakers.keys():
        if speaker_hash.upper() not in enrolled_hashes:
            unenrolled.append(speaker_hash)
    
    return jsonify({
        'success': True,
        'unenrolled': unenrolled,
        'enrolled': list(speaker_memory.keys())
    })


@app.route('/api/identify', methods=['POST'])
def identify_speaker():
    """Identify a speaker from audio."""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if not speaker_memory:
        return jsonify({'success': False, 'error': 'No speakers enrolled'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    threshold = float(request.form.get('threshold', 0.5))
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and process audio with noise reduction
        wav, sr = load_audio_with_noise_reduction(filepath, target_sr=16000, apply_nr=True)
        emb = extract_embedding(wav, sr)
        
        # Find best match
        best_name = "Unknown"
        best_score = -1.0
        all_scores = {}
        
        for name, ref_emb in speaker_memory.items():
            score = cosine_sim(emb, ref_emb)
            all_scores[name] = round(score, 4)
            if score > best_score:
                best_score = score
                best_name = name
        
        # Clean up
        os.remove(filepath)
        
        # Check threshold
        if best_score < threshold:
            identified = "Unknown"
        else:
            identified = best_name
        
        return jsonify({
            'success': True,
            'identified': identified,
            'best_match': best_name,
            'similarity': round(best_score, 4),
            'threshold': threshold,
            'all_scores': all_scores
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/speakers', methods=['GET'])
def list_speakers():
    """List all enrolled speakers."""
    return jsonify({
        'success': True,
        'speakers': list(speaker_memory.keys()),
        'count': len(speaker_memory)
    })


@app.route('/api/speakers/<name>', methods=['DELETE'])
def delete_speaker(name):
    """Delete an enrolled speaker."""
    if name in speaker_memory:
        del speaker_memory[name]
        save_speaker_memory()  # Persist change
        return jsonify({'success': True, 'message': f"Speaker '{name}' deleted"})
    return jsonify({'success': False, 'error': 'Speaker not found'}), 404


@app.route('/api/speakers/<name>/rename', methods=['POST'])
def rename_speaker(name):
    """Rename an enrolled speaker."""
    if name not in speaker_memory:
        return jsonify({'success': False, 'error': 'Speaker not found'}), 404
    
    new_name = request.json.get('new_name', '').strip() if request.json else ''
    if not new_name:
        return jsonify({'success': False, 'error': 'New name is required'}), 400
    
    if new_name in speaker_memory and new_name != name:
        return jsonify({'success': False, 'error': f"Speaker '{new_name}' already exists"}), 400
    
    # Move embedding to new name
    speaker_memory[new_name] = speaker_memory.pop(name)
    save_speaker_memory()  # Persist change
    
    return jsonify({
        'success': True,
        'message': f"Speaker renamed from '{name}' to '{new_name}'",
        'old_name': name,
        'new_name': new_name
    })


@app.route('/api/speakers/debug', methods=['GET'])
def speakers_debug():
    """Get detailed debug info about enrolled speakers including embedding hashes."""
    debug_info = {}
    for name, emb in speaker_memory.items():
        # Handle both numpy arrays and PyTorch tensors
        if hasattr(emb, 'cpu'):
            emb_np = emb.cpu().numpy()
        else:
            emb_np = emb
        debug_info[name] = {
            'hash': embedding_to_hash(emb),
            'embedding_size': emb_np.shape[0] if hasattr(emb_np, 'shape') else len(emb_np),
            'norm': float(np.linalg.norm(emb_np)),
            'sample': [round(float(x), 4) for x in emb_np[:5]]  # First 5 values
        }
    
    return jsonify({
        'success': True,
        'speakers': debug_info,
        'count': len(speaker_memory)
    })


@app.route('/api/clear', methods=['POST'])
def clear_speakers():
    """Clear all enrolled speakers."""
    speaker_memory.clear()
    return jsonify({'success': True, 'message': 'All speakers cleared'})


@app.route('/api/identify-live', methods=['POST'])
def identify_live():
    """Identify speaker from live audio chunk (WebM/Opus from browser or WAV).
    
    Now supports session-based speaker fingerprinting (like YouTube tab):
    - If enrolled match found above threshold: returns enrolled name
    - If no match: assigns/matches a session speaker hash (e.g., #DDF423)
    """
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    threshold = float(request.form.get('threshold', 0.5))
    noise_level = int(request.form.get('noise_level', 80))
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Auto-detect format from filename
        audio_format = 'webm'
        if audio_file.filename:
            if audio_file.filename.endswith('.wav'):
                audio_format = 'wav'
        
        if audio_format == 'wav':
            # Direct WAV - just load with librosa
            wav, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        else:
            # Convert WebM/Opus to WAV using pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Export to WAV bytes
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Load with librosa
            wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        
        # Apply noise reduction
        wav = reduce_noise(wav, sr, level=noise_level)
        
        # Check if audio is too short or silent
        if len(wav) < 1600:  # Less than 0.1 second
            return jsonify({'success': True, 'identified': 'Unknown', 'similarity': 0, 'message': 'Audio too short'})
        
        # Check for silence
        if np.max(np.abs(wav)) < 0.01:
            return jsonify({'success': True, 'identified': 'Unknown', 'similarity': 0, 'message': 'No audio detected'})
        
        # Extract embedding
        emb = extract_embedding(wav, sr)
        
        # Find best match among enrolled speakers
        best_name = None
        best_score = -1.0
        all_scores = {}
        
        for name, ref_emb in speaker_memory.items():
            score = cosine_sim(emb, ref_emb)
            all_scores[name] = round(score, 4)
            print(f"  Speaker '{name}': similarity = {score:.4f}")
            if score > best_score:
                best_score = score
                best_name = name
        
        # Check if we have an enrolled match above threshold
        if best_name and best_score >= threshold:
            identified = best_name
            print(f"  -> IDENTIFIED (enrolled): {identified} @ {best_score:.4f}")
        else:
            # No enrolled match - use session speaker fingerprinting (like YouTube tab)
            speaker_hash = get_stable_speaker_id(emb)
            identified = f"#{speaker_hash}"
            print(f"  -> FINGERPRINTED: {identified} (best enrolled: {best_name} @ {best_score:.4f}, threshold: {threshold})")
        
        return jsonify({
            'success': True,
            'identified': identified,
            'best_match': best_name or 'None',
            'similarity': round(best_score, 4) if best_score >= 0 else 0,
            'all_scores': all_scores
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/enroll-live', methods=['POST'])
def enroll_live():
    """Enroll a speaker from live audio (WebM/Opus from browser or WAV)."""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Speaker name is required'}), 400
    
    noise_level = int(request.form.get('noise_level', 80))
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Auto-detect format from filename
        audio_format = 'webm'
        if audio_file.filename:
            if audio_file.filename.endswith('.wav'):
                audio_format = 'wav'
        
        if audio_format == 'wav':
            # Direct WAV - just load with librosa
            wav, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        else:
            # Convert WebM/Opus to WAV using pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Export to WAV bytes
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Load with librosa
            wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        
        # Apply noise reduction
        wav = reduce_noise(wav, sr, level=noise_level)
        
        # Check audio quality
        duration = len(wav) / sr
        if duration < 1.0:
            return jsonify({'success': False, 'error': f'Recording too short ({duration:.1f}s). Please record at least 2 seconds.'}), 400
        
        # Check for silence
        if np.max(np.abs(wav)) < 0.01:
            return jsonify({'success': False, 'error': 'No audio detected. Please speak louder.'}), 400
        
        # Extract embedding
        emb = extract_embedding(wav, sr)
        
        # Store embedding
        speaker_memory[name] = emb
        print(f"Enrolled speaker '{name}' with embedding shape: {emb.shape}")
        
        # Also add to session speakers so future chunks match by enrolled name
        if hasattr(emb, 'cpu'):
            session_emb = emb.cpu().numpy().flatten()
        else:
            session_emb = np.array(emb).flatten()
        session_speakers[name] = session_emb
        session_speaker_counts[name] = 1
        
        # Find matching session speakers to retroactively update
        matching_hashes = []
        if hasattr(emb, 'cpu'):
            enrolled_np = emb.cpu().numpy().flatten()
        else:
            enrolled_np = np.array(emb).flatten()
        
        for session_hash, session_emb in session_speakers.items():
            sim = float(np.dot(enrolled_np, session_emb.flatten()) / 
                       (np.linalg.norm(enrolled_np) * np.linalg.norm(session_emb) + 1e-8))
            if sim >= 0.20:  # Same threshold as clustering
                matching_hashes.append(session_hash)
                print(f"Enrolled '{name}' matches session speaker #{session_hash} (sim={sim:.3f})")
        
        # Save to disk for persistence
        save_speaker_memory()
        
        return jsonify({
            'success': True,
            'message': f"Speaker '{name}' enrolled successfully",
            'duration': round(duration, 1),
            'speaker_count': len(speaker_memory),
            'matching_hashes': matching_hashes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using the currently loaded ASR model (Parakeet or Voxtral)."""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    # Check if any model is loaded
    if parakeet_model is None and voxtral_model is None:
        return jsonify({
            'success': False, 
            'error': 'No ASR model loaded. Please select and initialize a model in Settings.',
            'use_browser': True
        }), 400
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        noise_level = int(request.form.get('noise_level', 80))
        audio_format = request.form.get('format', 'webm')  # Support wav or webm
        
        # Auto-detect format from filename
        if audio_file.filename:
            if audio_file.filename.endswith('.wav'):
                audio_format = 'wav'
            elif audio_file.filename.endswith('.webm'):
                audio_format = 'webm'
        
        # Save to temp file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_transcribe.wav')
        
        if audio_format == 'wav':
            # Direct WAV - just load with librosa
            wav, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        else:
            # Convert WebM to WAV using pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            audio_segment.export(temp_path, format="wav")
            wav, sr = librosa.load(temp_path, sr=16000, mono=True)
        
        # Apply noise reduction
        wav = reduce_noise(wav, sr, level=noise_level)
        sf.write(temp_path, wav, sr)
        
        # Transcribe with whichever model is loaded (Parakeet preferred for speed)
        text = None
        model_used = None
        
        if parakeet_model is not None:
            text = transcribe_with_parakeet(temp_path)
            model_used = current_parakeet_model_name
        elif voxtral_model is not None:
            text = transcribe_with_voxtral(temp_path)
            model_used = current_voxtral_model_name
        
        print(f"Transcription text (using {model_used}): '{text}'")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if text:
            return jsonify({
                'success': True,
                'text': text,
                'model': model_used,
                'use_browser': False
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Transcription returned empty result',
                'use_browser': True
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reset-session-speakers', methods=['POST'])
def reset_session_speakers():
    """Reset session speaker clustering for a fresh start."""
    global session_speakers, session_speaker_counts, accumulated_audio, chunk_history, last_stable_speaker, chunk_index_counter
    session_speakers = {}
    session_speaker_counts = {}
    accumulated_audio = []  # Reset audio accumulation
    chunk_history = []  # Reset chunk history
    last_stable_speaker = None
    chunk_index_counter = 0
    print("Session speakers, audio buffer, and chunk history reset")
    return jsonify({'success': True})


@app.route('/api/process-streaming', methods=['POST'])
def process_streaming():
    """Process small audio chunk for streaming transcription with speaker detection."""
    if not model_loaded:
        print("ERROR: Model not initialized")
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        print(f"ERROR: No audio in request. Files: {list(request.files.keys())}")
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    threshold = float(request.form.get('threshold', 0.5))
    noise_level = int(request.form.get('noise_level', 80))
    prev_embedding_json = request.form.get('prev_embedding', None)
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Auto-detect format
        audio_format = 'webm'
        if audio_file.filename and audio_file.filename.endswith('.wav'):
            audio_format = 'wav'
        
        if audio_format == 'wav':
            wav, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        else:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        
        # Apply noise reduction
        wav = reduce_noise(wav, sr, level=noise_level)
        
        # Check for valid audio
        if len(wav) < 1600 or np.max(np.abs(wav)) < 0.01:
            return jsonify({
                'success': True,
                'text': '',
                'speaker': 'Unknown',
                'is_sentence_end': False,
                'speaker_changed': False
            })
        
        # Use lock for GPU operations to prevent race condition
        with processing_lock:
            # Save for transcription (fast, small chunk)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_stream.wav')
            sf.write(temp_path, wav, sr)
            
            # Transcribe with whichever model is loaded
            text = None
            if parakeet_model is not None:
                text = transcribe_with_parakeet(temp_path)
            elif voxtral_model is not None:
                text = transcribe_with_voxtral(temp_path)
            if not text:
                text = ''
            
            # Always extract per-chunk embedding for history tracking
            chunk_embedding = extract_embedding(wav, sr)
            
            # Accumulate audio for speaker ID (need longer chunks for stability)
            global accumulated_audio, chunk_history, last_stable_speaker, chunk_index_counter
            accumulated_audio.append(wav)
            
            # Store in chunk history
            chunk_index_counter += 1
            if hasattr(chunk_embedding, 'cpu'):
                chunk_emb_np = chunk_embedding.cpu().numpy().flatten()
            else:
                chunk_emb_np = np.array(chunk_embedding).flatten()
            
            chunk_history.append({
                'embedding': chunk_emb_np.copy(),
                'text': text,
                'index': chunk_index_counter
            })
            # Keep only last N chunks
            if len(chunk_history) > MAX_CHUNK_HISTORY:
                chunk_history.pop(0)
            
            # IMMEDIATE chunk-level speaker change detection (before accumulation)
            immediate_speaker_change = False
            if prev_embedding_json:
                try:
                    prev_emb = np.array(json.loads(prev_embedding_json))
                    # Compare current chunk directly with previous embedding
                    chunk_vs_prev_sim = float(np.dot(chunk_emb_np.flatten(), prev_emb.flatten()) / 
                                              (np.linalg.norm(chunk_emb_np) * np.linalg.norm(prev_emb) + 1e-8))
                    if chunk_vs_prev_sim < 0.20:  # Very low similarity = likely different speaker (was 0.35, too aggressive)
                        immediate_speaker_change = True
                        print(f"IMMEDIATE speaker change detected: chunk_sim={chunk_vs_prev_sim:.3f}")
                except:
                    pass
            
            # Calculate total accumulated duration
            total_samples = sum(len(chunk) for chunk in accumulated_audio)
            total_duration = total_samples / sr
            
            # Variables for response
            stable_embedding_ready = False
            speaker_change_detected = False
            split_at_index = None
            
            # If immediate speaker change detected, reset accumulation for faster ID
            if immediate_speaker_change:
                # Clear old accumulated audio - start fresh with new speaker
                accumulated_audio = [wav]  # Start fresh with current chunk
                print("Accumulation reset due to immediate speaker change")
            
            # Only do stable speaker detection when we have enough audio
            if total_duration >= ACCUM_DURATION_FOR_SPEAKER_ID:
                # Concatenate accumulated audio
                accumulated_wav = np.concatenate(accumulated_audio)
                current_embedding = extract_embedding(accumulated_wav, sr)
                stable_embedding_ready = True
                
                # Keep last 2 seconds for continuity (sliding window)
                keep_samples = int(2.0 * sr)
                if len(accumulated_wav) > keep_samples:
                    accumulated_audio = [accumulated_wav[-keep_samples:]]
                else:
                    accumulated_audio = [accumulated_wav]
                print(f"Speaker ID with {total_duration:.1f}s accumulated audio")
            else:
                # Not enough audio yet - use chunk embedding
                current_embedding = chunk_embedding
                print(f"Accumulating audio: {total_duration:.1f}s / {ACCUM_DURATION_FOR_SPEAKER_ID}s")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Check for sentence end
        is_sentence_end = bool(text) and text.rstrip().endswith(('.', '?', '!', '."', '?"', '!"'))
        
        # Convert to numpy for comparison and JSON
        if hasattr(current_embedding, 'cpu'):
            current_emb_np = current_embedding.cpu().numpy()
        else:
            current_emb_np = current_embedding
        
        # Identify speaker (use stable clustered ID if no enrolled match)
        stable_speaker_id = get_stable_speaker_id(current_emb_np)
        # Check if stable_speaker_id is an enrolled name (no # prefix) or a hash (needs # prefix)
        if stable_speaker_id in speaker_memory:
            speaker = stable_speaker_id  # Enrolled name
        else:
            speaker = f'#{stable_speaker_id}'  # Hash
        best_sim = 0
        matched_enrolled_name = None
        if speaker_memory:
            for name, emb in speaker_memory.items():
                if hasattr(emb, 'cpu'):
                    emb_np = emb.cpu().numpy()
                else:
                    emb_np = emb
                sim = float(np.dot(current_emb_np.flatten(), emb_np.flatten()) / 
                           (np.linalg.norm(current_emb_np) * np.linalg.norm(emb_np) + 1e-8))
                if sim > best_sim and sim > threshold:
                    best_sim = sim
                    speaker = name  # Use enrolled name if matched
                    matched_enrolled_name = name
        
        # Adaptive profile update: improve enrolled speaker embedding with new audio
        if matched_enrolled_name and best_sim >= ADAPTIVE_UPDATE_THRESHOLD and stable_embedding_ready:
            update_speaker_profile(matched_enrolled_name, current_emb_np)
        
        # HYBRID SPEAKER CHANGE DETECTION
        # Use Pyannote for immediate change detection (every chunk)
        # Use ECAPA-TDNN 3-second accumulation for stable speaker ID
        speaker_changed = False
        split_at_chunk_index = None
        change_method = 'none'
        
        # Check for immediate speaker change first (from chunk-level detection)
        if immediate_speaker_change:
            speaker_changed = True
            change_method = 'immediate_chunk'
            print(f"Speaker change flagged: method=immediate_chunk")
        
        # Get previous embedding for comparison
        prev_emb = None
        if prev_embedding_json:
            try:
                prev_emb = np.array(json.loads(prev_embedding_json))
            except:
                pass
        
        # Try hybrid detection on EVERY chunk (not just after 3 seconds)
        print(f"Hybrid check: pyannote={pyannote_segmentation is not None}, prev_emb={prev_emb is not None}")
        if pyannote_segmentation is not None or prev_emb is not None:
            hybrid_result = hybrid_speaker_change_detection(
                wav, sr, 
                current_emb_np, 
                prev_emb,
                ecapa_threshold=0.30  # Threshold for speaker change (was 0.40, too aggressive for single speaker)
            )
            print(f"Hybrid result: changed={hybrid_result['speaker_changed']}, method={hybrid_result['method']}, ecapa_sim={hybrid_result['ecapa_sim']:.3f}")
            if hybrid_result['speaker_changed']:
                speaker_changed = True
                change_method = hybrid_result['method']
                print(f"Hybrid change detected: method={change_method}, confidence={hybrid_result['confidence']:.2f}")
                print(f"  ECAPA sim: {hybrid_result['ecapa_sim']:.3f}, Pyannote prob: {hybrid_result['pyannote_prob']:.3f}")
        
        # Also check stable speaker change (backup for when hybrid misses)
        if stable_embedding_ready:
            if last_stable_speaker is not None and speaker != last_stable_speaker:
                if not speaker_changed:  # Only if hybrid didn't already detect
                    speaker_changed = True
                    change_method = 'stable_3s'
                    print(f"Stable speaker change: {last_stable_speaker} -> {speaker}")
                
                # Find exact split point by checking chunk history
                if len(chunk_history) > 1:
                    new_speaker_emb = current_emb_np.flatten()
                    for i, chunk in enumerate(chunk_history[:-1]):
                        chunk_emb = chunk['embedding']
                        sim = float(np.dot(chunk_emb, new_speaker_emb) / 
                                   (np.linalg.norm(chunk_emb) * np.linalg.norm(new_speaker_emb) + 1e-8))
                        if sim > 0.20:
                            split_at_chunk_index = chunk['index']
                            print(f"Split point found at chunk {split_at_chunk_index} (sim={sim:.3f})")
                            break
            
            # Update last stable speaker
            last_stable_speaker = speaker
        
        return jsonify({
            'success': True,
            'text': text,
            'speaker': speaker,
            'similarity': round(best_sim, 3),
            'is_sentence_end': is_sentence_end,
            'speaker_changed': speaker_changed,
            'change_method': change_method,
            'split_at_chunk_index': split_at_chunk_index,
            'chunk_index': chunk_index_counter,
            'embedding': current_emb_np.flatten().tolist()  # For next comparison
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-enroll', methods=['POST'])
def auto_enroll():
    """
    Auto-enroll a speaker based on spoken name.
    Expects: audio file and transcript (from browser speech recognition or OpenAI)
    """
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    transcript = request.form.get('transcript', '').strip()
    if not transcript:
        return jsonify({'success': False, 'error': 'No transcript provided'}), 400
    
    # Extract name from transcript
    name = extract_name_from_text(transcript)
    if not name:
        return jsonify({
            'success': False,
            'error': 'Could not detect a name. Please say something like "My name is [Your Name]"',
            'transcript': transcript
        }), 400
    
    noise_level = int(request.form.get('noise_level', 80))
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Convert WebM/Opus to WAV
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Load with librosa and apply noise reduction
        wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        wav = reduce_noise(wav, sr, level=noise_level)
        
        # Check audio quality
        duration = len(wav) / sr
        if duration < 1.0:
            return jsonify({'success': False, 'error': f'Recording too short ({duration:.1f}s)'}), 400
        
        if np.max(np.abs(wav)) < 0.01:
            return jsonify({'success': False, 'error': 'No audio detected'}), 400
        
        # Extract embedding and store
        emb = extract_embedding(wav, sr)
        speaker_memory[name] = emb
        
        return jsonify({
            'success': True,
            'message': f"Speaker '{name}' enrolled successfully",
            'name': name,
            'transcript': transcript,
            'duration': round(duration, 1),
            'speaker_count': len(speaker_memory)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# YouTube Processing
# =============================================================================

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
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(wav) / sr
        
        # Split into segments
        segment_samples = int(segment_duration * sr)
        num_segments = int(np.ceil(len(wav) / segment_samples))
        
        transcript = []
        speaker_counts = {}
        
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
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'yt_seg_{job_id}_{i}.wav')
                sf.write(temp_path, segment, sr)
                text = transcribe_with_parakeet(temp_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if not text or text.strip() == '':
                continue
            
            # Identify speaker
            speaker = 'Unknown'
            if model_loaded and speaker_memory:
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


def extract_youtube_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    import re
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@app.route('/api/youtube/download', methods=['POST'])
def youtube_download():
    """Download YouTube audio only (no automatic processing). Uses cached file if available."""
    if not YTDLP_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'yt-dlp not installed. Run: pip install yt-dlp'
        }), 400
    
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    
    # Extract video ID for caching
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return jsonify({'success': False, 'error': 'Could not extract video ID from URL'}), 400
    
    # Use video ID as job ID for consistent file naming
    job_id = video_id
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'youtube_{video_id}.wav')
    
    try:
        # Check if file already exists (cached)
        if os.path.exists(output_path):
            print(f"Using cached audio for video: {video_id}")
            # Get video info without downloading
            ydl_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                duration_str = f"{duration // 60}:{duration % 60:02d}" if duration else "Unknown"
            
            audio_url = f'/api/youtube/audio/{job_id}'
            youtube_jobs[job_id] = {
                'status': 'ready',
                'title': title,
                'duration': duration_str,
                'audio_path': output_path,
                'audio_url': audio_url
            }
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'title': title,
                'duration': duration_str,
                'audio_url': audio_url,
                'cached': True
            })
        
        # Download audio (not cached)
        print(f"Downloading audio for video: {video_id}")
        info = download_youtube_audio(url, output_path)
        
        # Initialize job (download only, no processing)
        audio_url = f'/api/youtube/audio/{job_id}'
        
        youtube_jobs[job_id] = {
            'status': 'ready',
            'title': info['title'],
            'duration': info['duration_str'],
            'audio_path': output_path,
            'audio_url': audio_url
        }
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'title': info['title'],
            'duration': info['duration_str'],
            'audio_url': audio_url,
            'cached': False
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/youtube/process', methods=['POST'])
def youtube_process():
    """Start processing a YouTube video."""
    if not YTDLP_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'yt-dlp not installed. Run: pip install yt-dlp'
        }), 400
    
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Download audio
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'youtube_{job_id}.wav')
        info = download_youtube_audio(url, output_path)
        
        # Initialize job with audio URL
        audio_url = f'/api/youtube/audio/{job_id}'
        
        youtube_jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting...',
            'transcript': [],
            'speakers': [],
            'title': info['title'],
            'duration': info['duration_str'],
            'audio_path': output_path,
            'audio_url': audio_url
        }
        
        # Start processing in background
        thread = threading.Thread(
            target=process_youtube_audio,
            args=(job_id, output_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'title': info['title'],
            'duration': info['duration_str'],
            'audio_url': audio_url
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/youtube/audio/<job_id>', methods=['GET'])
def youtube_audio(job_id):
    """Serve YouTube audio file with CORS headers for Web Audio API."""
    if job_id not in youtube_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    audio_path = youtube_jobs[job_id].get('audio_path')
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    from flask import send_file, make_response
    response = make_response(send_file(audio_path, mimetype='audio/wav'))
    # Add CORS headers for Web Audio API access
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/api/youtube/status/<job_id>', methods=['GET'])
def youtube_status(job_id):
    """Get status of a YouTube processing job."""
    if job_id not in youtube_jobs:
        return jsonify({'status': 'error', 'error': 'Job not found'}), 404
    
    job = youtube_jobs[job_id]
    return jsonify(job)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Starting Speaker Identification Web Server...")
    print(f"Device: {device}")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
