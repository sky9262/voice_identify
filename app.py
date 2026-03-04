# -*- coding: utf-8 -*-
"""
Speaker Identification Web Server

Using SpeechBrain ECAPA-TDNN for high-accuracy speaker verification.
"""

import os
import io
import re
import uuid
import threading
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
try:
    from parakeet_mlx import from_pretrained as parakeet_from_pretrained
    PARAKEET_AVAILABLE = True
    print("Parakeet-MLX loaded (Apple Silicon GPU optimized)")
except ImportError as e:
    print(f"Warning: parakeet-mlx import failed: {e}")
except Exception as e:
    print(f"Warning: parakeet-mlx error: {e}")

# =============================================================================
# Flask App Setup
# =============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

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


def embedding_to_hash(embedding, length=8):
    """Convert embedding to short hex hash for human-readable identification."""
    import hashlib
    emb_bytes = embedding.astype(np.float32).tobytes()
    full_hash = hashlib.sha256(emb_bytes).hexdigest()
    return full_hash[:length].upper()


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


def init_parakeet():
    """Initialize Parakeet-MLX model."""
    global parakeet_model, PARAKEET_AVAILABLE
    
    if not PARAKEET_AVAILABLE:
        return False
    
    if parakeet_model is not None:
        return True
    
    try:
        print("Loading Parakeet-MLX model (mlx-community/parakeet-tdt-0.6b-v3)...")
        # Load the TDT model - more accurate than CTC
        parakeet_model = parakeet_from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        print("Parakeet-MLX TDT model loaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to load Parakeet-MLX model: {e}")
        import traceback
        traceback.print_exc()
        PARAKEET_AVAILABLE = False
        return False


def transcribe_with_parakeet(audio_path):
    """Transcribe audio using Parakeet-MLX (Apple Silicon optimized)."""
    global parakeet_model
    
    if not PARAKEET_AVAILABLE:
        print("Parakeet not available")
        return None
    
    if parakeet_model is None:
        if not init_parakeet():
            return None
    
    try:
        # Use parakeet-mlx transcribe
        print(f"Transcribing: {audio_path}")
        result = parakeet_model.transcribe(audio_path)
        print(f"Raw transcription result: {result}, type: {type(result)}")
        
        # Handle different return formats
        if result is None:
            return None
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
    
    return jsonify({
        'model_loaded': model_loaded,
        'speaker_model': 'ECAPA-TDNN' if model_loaded else 'Not loaded',
        'parakeet_available': PARAKEET_AVAILABLE,
        'parakeet_loaded': parakeet_model is not None,
        'device': device,
        'speakers_enrolled': speakers_with_hash,
        'speaker_count': len(speaker_memory)
    })


@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize the model."""
    try:
        init_model()
        # Also init Parakeet for transcription
        if PARAKEET_AVAILABLE:
            init_parakeet()
        return jsonify({'success': True, 'message': 'Model initialized successfully'})
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
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and process audio with noise reduction
        wav, sr = load_audio_with_noise_reduction(filepath, target_sr=16000, apply_nr=True)
        emb = extract_embedding(wav, sr)
        
        # Store embedding
        speaker_memory[name] = emb
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f"Speaker '{name}' enrolled successfully",
            'speaker_count': len(speaker_memory)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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
        debug_info[name] = {
            'hash': embedding_to_hash(emb),
            'embedding_size': emb.shape[0] if hasattr(emb, 'shape') else len(emb),
            'norm': float(np.linalg.norm(emb)),
            'sample': [round(float(x), 4) for x in emb[:5]]  # First 5 values
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
    """Identify speaker from live audio chunk (WebM/Opus from browser or WAV)."""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if not speaker_memory:
        return jsonify({'success': False, 'error': 'No speakers enrolled'}), 400
    
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
        
        # Find best match
        best_name = "Unknown"
        best_score = -1.0
        all_scores = {}
        
        for name, ref_emb in speaker_memory.items():
            score = cosine_sim(emb, ref_emb)
            all_scores[name] = round(score, 4)
            print(f"  Speaker '{name}': similarity = {score:.4f}")
            if score > best_score:
                best_score = score
                best_name = name
        
        # Check threshold
        if best_score < threshold:
            identified = "Unknown"
            print(f"  -> UNKNOWN (best: {best_name} @ {best_score:.4f}, threshold: {threshold})")
        else:
            identified = best_name
            print(f"  -> IDENTIFIED: {identified} @ {best_score:.4f}")
        
        return jsonify({
            'success': True,
            'identified': identified,
            'best_match': best_name,
            'similarity': round(best_score, 4),
            'all_scores': all_scores
        })
    except Exception as e:
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
        
        return jsonify({
            'success': True,
            'message': f"Speaker '{name}' enrolled successfully",
            'duration': round(duration, 1),
            'speaker_count': len(speaker_memory)
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
    """Transcribe audio using Parakeet-MLX (Apple Silicon optimized)."""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    if not PARAKEET_AVAILABLE:
        return jsonify({
            'success': False, 
            'error': 'Parakeet-MLX not available. Install with: pip install parakeet-mlx',
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
        
        # Transcribe with Parakeet
        text = transcribe_with_parakeet(temp_path)
        print(f"Transcription text: '{text}'")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if text:
            return jsonify({
                'success': True,
                'text': text,
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
    app.run(debug=True, host='0.0.0.0', port=5001)
