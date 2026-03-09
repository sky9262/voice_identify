# -*- coding: utf-8 -*-
"""
Model loading and inference for Speaker Identification System.
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F

import logger as log
from config import device, ECAPA_MODEL_ID

# Fix torchaudio compatibility for SpeechBrain
try:
    import torchaudio
    # SpeechBrain may need this attribute
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile']
except ImportError:
    pass

# =============================================================================
# Model Availability Flags
# =============================================================================

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

# Parakeet-MLX (Apple Silicon optimized)
PARAKEET_AVAILABLE = False
parakeet_from_pretrained = None
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
voxtral_load_model = None
voxtral_generate = None
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
PyannoteModel = None
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
# Global Model State
# =============================================================================

ecapa_model = None  # SpeechBrain ECAPA-TDNN
parakeet_model = None
voxtral_model = None
pyannote_segmentation = None
current_parakeet_model_name = "mlx-community/parakeet-tdt-1.1b"  # Track current model
current_voxtral_model_name = None
model_loaded = False

# =============================================================================
# Model Initialization
# =============================================================================

def init_model():
    """Initialize SpeechBrain ECAPA-TDNN model."""
    global ecapa_model, model_loaded
    
    if model_loaded:
        return True
    
    if not SPEECHBRAIN_AVAILABLE:
        log.error("SpeechBrain not installed! Install with: pip install speechbrain")
        return False
    
    log.info(f"Loading ECAPA-TDNN model on {device}...")
    
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
                log.info("ECAPA-TDNN loaded on MPS (Apple Silicon GPU)")
            except Exception as mps_error:
                log.warning(f"MPS failed ({mps_error}), falling back to CPU")
                ecapa_model = EncoderClassifier.from_hparams(
                    source=ECAPA_MODEL_ID,
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"}
                )
                log.info("ECAPA-TDNN loaded on CPU")
        else:
            ecapa_model = EncoderClassifier.from_hparams(
                source=ECAPA_MODEL_ID,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
            log.info(f"ECAPA-TDNN loaded on {device}")
        
        model_loaded = True
        log.info("ECAPA-TDNN speaker verification ready (EER ~0.8%)")
        return True
        
    except Exception as e:
        log.error(f"Failed to load ECAPA-TDNN: {e}")
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


# =============================================================================
# Transcription Functions
# =============================================================================

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


# =============================================================================
# Speaker Embedding Functions
# =============================================================================

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


# =============================================================================
# Speaker Change Detection
# =============================================================================

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


# =============================================================================
# Getters for model state
# =============================================================================

def is_model_loaded():
    """Check if speaker model is loaded."""
    return model_loaded


def is_parakeet_loaded():
    """Check if Parakeet model is loaded."""
    return parakeet_model is not None


def is_voxtral_loaded():
    """Check if Voxtral model is loaded."""
    return voxtral_model is not None


def is_pyannote_loaded():
    """Check if Pyannote model is loaded."""
    return pyannote_segmentation is not None


def get_current_asr_model_name():
    """Get the name of currently loaded ASR model."""
    if parakeet_model is not None:
        return current_parakeet_model_name
    elif voxtral_model is not None:
        return current_voxtral_model_name
    return None
