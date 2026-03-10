# -*- coding: utf-8 -*-
"""
API routes for Speaker Identification System.
"""

import os
import io
import json
import uuid
import threading
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
from flask import Blueprint, request, jsonify, send_file, make_response
from werkzeug.utils import secure_filename
from pydub import AudioSegment

from config import (
    UPLOAD_FOLDER,
    PARAKEET_MODELS,
    VOXTRAL_MODELS,
    processing_lock,
    device,
    ADAPTIVE_UPDATE_THRESHOLD,
    ACCUM_DURATION_FOR_SPEAKER_ID
)
from utils import (
    allowed_file,
    embedding_to_hash,
    convert_webm_to_wav,
    extract_name_from_text,
    extract_youtube_video_id,
    sanitize_speaker_name
)
from audio import (
    reduce_noise,
    load_audio,
    load_audio_with_noise_reduction,
    load_audio_from_bytes,
    is_audio_valid
)
from models import (
    init_model,
    init_pyannote_segmentation,
    init_parakeet,
    switch_parakeet_model,
    switch_voxtral_model,
    transcribe_with_parakeet,
    transcribe_with_voxtral,
    transcribe_audio,
    extract_embedding,
    cosine_sim,
    hybrid_speaker_change_detection,
    is_model_loaded,
    is_parakeet_loaded,
    is_voxtral_loaded,
    is_pyannote_loaded,
    get_current_asr_model_name,
    SPEECHBRAIN_AVAILABLE,
    PARAKEET_AVAILABLE,
    VOXTRAL_AVAILABLE,
    PYANNOTE_AVAILABLE,
    VIBEVOICE_AVAILABLE,
    parakeet_model,
    voxtral_model,
    pyannote_segmentation,
    current_parakeet_model_name,
    current_voxtral_model_name,
    init_vibevoice,
    transcribe_with_vibevoice,
    is_vibevoice_loaded,
    get_vibevoice_status
)
from speakers import (
    get_speaker_memory,
    get_session_speakers,
    get_session_speaker_counts,
    add_speaker,
    remove_speaker,
    rename_speaker,
    clear_all_speakers,
    update_speaker_profile,
    get_stable_speaker_id,
    reset_session,
    add_to_session_speakers,
    get_accumulated_audio,
    add_audio_chunk,
    set_accumulated_audio,
    get_accumulated_duration,
    get_chunk_history,
    add_chunk_to_history,
    get_chunk_index,
    get_last_stable_speaker,
    set_last_stable_speaker,
    identify_speaker,
    enroll_from_session_hash,
    save_speaker_memory
)
from youtube import (
    download_youtube_audio,
    process_youtube_audio,
    get_youtube_jobs,
    get_youtube_job,
    set_youtube_job,
    is_ytdlp_available,
    YTDLP_AVAILABLE
)

# Create blueprint
api = Blueprint('api', __name__)

# =============================================================================
# Session Audio Recording (for VibeVoice post-processing)
# =============================================================================

session_audio_chunks = []  # List of (audio_data, timestamp) tuples
session_audio_sample_rate = 16000
session_start_time = None


# =============================================================================
# Model Management Routes
# =============================================================================

@api.route('/switch-model', methods=['POST'])
def api_switch_model():
    """Switch to a different ASR model (Parakeet or Voxtral)."""
    data = request.get_json()
    model_name = data.get('model') if data else None
    print(f"API received model switch request: {model_name}")  # Debug
    
    if not model_name:
        return jsonify({'success': False, 'error': 'No model specified'}), 400
    
    all_valid_models = PARAKEET_MODELS + VOXTRAL_MODELS
    
    if model_name not in all_valid_models:
        return jsonify({'success': False, 'error': f'Invalid model. Valid options: {all_valid_models}'}), 400
    
    # Route to appropriate model loader
    if model_name in PARAKEET_MODELS:
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


@api.route('/status', methods=['GET'])
def status():
    """Get system status."""
    speaker_memory = get_speaker_memory()
    
    # Build speaker list with hashes
    speakers_with_hash = []
    for name, emb in speaker_memory.items():
        speakers_with_hash.append({
            'name': name,
            'hash': embedding_to_hash(emb)
        })
    
    # Determine active model name
    active_model = get_current_asr_model_name()
    
    return jsonify({
        'model_loaded': is_model_loaded(),
        'speaker_model': 'ECAPA-TDNN' if is_model_loaded() else 'Not loaded',
        'parakeet_available': PARAKEET_AVAILABLE,
        'parakeet_loaded': is_parakeet_loaded(),
        'parakeet_model_name': current_parakeet_model_name if is_parakeet_loaded() else None,
        'voxtral_available': VOXTRAL_AVAILABLE,
        'voxtral_loaded': is_voxtral_loaded(),
        'voxtral_model_name': current_voxtral_model_name if is_voxtral_loaded() else None,
        'asr_model_loaded': is_parakeet_loaded() or is_voxtral_loaded(),
        'asr_model_name': active_model,
        'pyannote_available': PYANNOTE_AVAILABLE,
        'pyannote_loaded': is_pyannote_loaded(),
        'vibevoice_available': VIBEVOICE_AVAILABLE,
        'vibevoice_loaded': is_vibevoice_loaded(),
        'device': device,
        'speakers_enrolled': speakers_with_hash,
        'speaker_count': len(speaker_memory)
    })


@api.route('/init', methods=['POST'])
def initialize():
    """Initialize the model."""
    try:
        init_model()
        # Don't auto-init Parakeet - let user choose model in Settings
        return jsonify({'success': True, 'message': 'Model initialized successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/init-pyannote', methods=['POST'])
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


@api.route('/init-vibevoice', methods=['POST'])
def initialize_vibevoice():
    """Initialize VibeVoice-ASR model for post-processing."""
    if not VIBEVOICE_AVAILABLE:
        return jsonify({
            'success': False, 
            'error': 'VibeVoice not available - mlx-audio not installed'
        }), 400
    
    try:
        success = init_vibevoice()
        if success:
            return jsonify({
                'success': True,
                'message': 'VibeVoice-ASR model loaded successfully',
                'model': 'mlx-community/VibeVoice-ASR-4bit'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load VibeVoice-ASR model'
            }), 400
    except Exception as e:
        log.error(f"Error initializing VibeVoice: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Speaker Enrollment Routes
# =============================================================================

@api.route('/enroll', methods=['POST'])
def enroll_speaker():
    """Enroll a new speaker."""
    if not is_model_loaded():
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    raw_name = request.form.get('name', '')
    
    # Validate and sanitize speaker name
    name, error = sanitize_speaker_name(raw_name)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    
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
            wav_buffer = convert_webm_to_wav(audio_data)
            wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        else:
            # Save and load other formats directly
            filepath = os.path.join(UPLOAD_FOLDER, filename)
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
        add_speaker(name, emb)
        
        speaker_memory = get_speaker_memory()
        return jsonify({
            'success': True,
            'message': f"Speaker '{name}' enrolled successfully ({duration:.1f}s audio)",
            'speaker_count': len(speaker_memory)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/enroll-from-hash', methods=['POST'])
def enroll_from_hash():
    """Enroll a speaker using their session hash (fingerprint)."""
    if not is_model_loaded():
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    data = request.get_json()
    speaker_hash = data.get('hash', '').strip().upper()
    raw_name = data.get('name', '')
    merge_with_existing = data.get('merge', False)
    
    if not speaker_hash:
        return jsonify({'success': False, 'error': 'Speaker hash is required'}), 400
    
    # Validate and sanitize speaker name
    name, error = sanitize_speaker_name(raw_name)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    
    success, message, update_count = enroll_from_session_hash(speaker_hash, name, merge_with_existing)
    
    if success:
        speaker_memory = get_speaker_memory()
        return jsonify({
            'success': True,
            'message': message,
            'speaker_count': len(speaker_memory),
            'hash': speaker_hash.lstrip('#'),
            'name': name
        })
    else:
        return jsonify({'success': False, 'error': message}), 400


@api.route('/session-speakers', methods=['GET'])
def get_session_speakers_route():
    """Get list of unenrolled session speakers (hashes)."""
    speaker_memory = get_speaker_memory()
    session_speakers = get_session_speakers()
    
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


# =============================================================================
# Speaker Identification Routes
# =============================================================================

@api.route('/identify', methods=['POST'])
def identify_speaker_route():
    """Identify a speaker from audio."""
    if not is_model_loaded():
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    speaker_memory = get_speaker_memory()
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
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Load and process audio with noise reduction
        wav, sr = load_audio_with_noise_reduction(filepath, target_sr=16000, apply_nr=True)
        emb = extract_embedding(wav, sr)
        
        # Find best match
        identified, best_score, all_scores = identify_speaker(emb, threshold)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'identified': identified or "Unknown",
            'best_match': list(all_scores.keys())[list(all_scores.values()).index(max(all_scores.values()))] if all_scores else 'Unknown',
            'similarity': round(best_score, 4) if best_score >= 0 else 0,
            'threshold': threshold,
            'all_scores': all_scores
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/identify-live', methods=['POST'])
def identify_live():
    """Identify speaker from live audio chunk (WebM/Opus from browser or WAV).
    
    Now supports session-based speaker fingerprinting (like YouTube tab):
    - If enrolled match found above threshold: returns enrolled name
    - If no match: assigns/matches a session speaker hash (e.g., #DDF423)
    """
    if not is_model_loaded():
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
        
        wav, sr = load_audio_from_bytes(audio_data, audio_format, 16000, noise_level)
        
        # Check if audio is valid
        valid, reason = is_audio_valid(wav, sr)
        if not valid:
            return jsonify({'success': True, 'identified': 'Unknown', 'similarity': 0, 'message': reason})
        
        # Extract embedding
        emb = extract_embedding(wav, sr)
        
        # Find best match among enrolled speakers
        speaker_memory = get_speaker_memory()
        identified, best_score, all_scores = identify_speaker(emb, threshold)
        
        if identified:
            print(f"  -> IDENTIFIED (enrolled): {identified} @ {best_score:.4f}")
        else:
            # No enrolled match - use session speaker fingerprinting (like YouTube tab)
            speaker_hash = get_stable_speaker_id(emb)
            identified = f"#{speaker_hash}"
            print(f"  -> FINGERPRINTED: {identified} (best enrolled: {list(all_scores.keys())[0] if all_scores else 'None'} @ {best_score:.4f}, threshold: {threshold})")
        
        return jsonify({
            'success': True,
            'identified': identified,
            'best_match': list(all_scores.keys())[0] if all_scores else 'None',
            'similarity': round(best_score, 4) if best_score >= 0 else 0,
            'all_scores': all_scores
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/enroll-live', methods=['POST'])
def enroll_live():
    """Enroll a speaker from live audio (WebM/Opus from browser or WAV)."""
    if not is_model_loaded():
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    raw_name = request.form.get('name', '')
    
    # Validate and sanitize speaker name
    name, error = sanitize_speaker_name(raw_name)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    
    noise_level = int(request.form.get('noise_level', 80))
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Auto-detect format from filename
        audio_format = 'webm'
        if audio_file.filename:
            if audio_file.filename.endswith('.wav'):
                audio_format = 'wav'
        
        wav, sr = load_audio_from_bytes(audio_data, audio_format, 16000, noise_level)
        
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
        add_speaker(name, emb)
        print(f"Enrolled speaker '{name}' with embedding shape: {emb.shape}")
        
        # Also add to session speakers so future chunks match by enrolled name
        add_to_session_speakers(name, emb)
        
        # Find matching session speakers to retroactively update
        session_speakers = get_session_speakers()
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
        
        speaker_memory = get_speaker_memory()
        return jsonify({
            'success': True,
            'message': f"Speaker '{name}' enrolled successfully",
            'duration': round(duration, 1),
            'speaker_count': len(speaker_memory),
            'matching_hashes': matching_hashes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Speaker Management Routes
# =============================================================================

@api.route('/speakers', methods=['GET'])
def list_speakers():
    """List all enrolled speakers."""
    speaker_memory = get_speaker_memory()
    return jsonify({
        'success': True,
        'speakers': list(speaker_memory.keys()),
        'count': len(speaker_memory)
    })


@api.route('/speakers/<name>', methods=['DELETE'])
def delete_speaker(name):
    """Delete an enrolled speaker."""
    if remove_speaker(name):
        return jsonify({'success': True, 'message': f"Speaker '{name}' deleted"})
    return jsonify({'success': False, 'error': 'Speaker not found'}), 404


@api.route('/speakers/<name>/rename', methods=['POST'])
def rename_speaker_route(name):
    """Rename an enrolled speaker."""
    speaker_memory = get_speaker_memory()
    if name not in speaker_memory:
        return jsonify({'success': False, 'error': 'Speaker not found'}), 404
    
    raw_new_name = request.json.get('new_name', '') if request.json else ''
    
    # Validate and sanitize new speaker name
    new_name, error = sanitize_speaker_name(raw_new_name)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    
    success, error = rename_speaker(name, new_name)
    
    if success:
        return jsonify({
            'success': True,
            'message': f"Speaker renamed from '{name}' to '{new_name}'",
            'old_name': name,
            'new_name': new_name
        })
    else:
        return jsonify({'success': False, 'error': error}), 400


@api.route('/speakers/debug', methods=['GET'])
def speakers_debug():
    """Get detailed debug info about enrolled speakers including embedding hashes."""
    speaker_memory = get_speaker_memory()
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


@api.route('/speakers/export', methods=['GET'])
def export_speakers():
    """Export all enrolled speakers as downloadable JSON."""
    speaker_memory = get_speaker_memory()
    export_data = {
        'version': '1.0',
        'exported_at': datetime.now().isoformat(),
        'speakers': {}
    }
    
    for name, emb in speaker_memory.items():
        if hasattr(emb, 'cpu'):
            emb_np = emb.cpu().numpy()
        else:
            emb_np = np.array(emb)
        export_data['speakers'][name] = {
            'embedding': emb_np.flatten().tolist(),
            'hash': embedding_to_hash(emb_np)
        }
    
    return jsonify({
        'success': True,
        'data': export_data,
        'count': len(speaker_memory)
    })


@api.route('/speakers/import', methods=['POST'])
def import_speakers():
    """Import speakers from JSON data."""
    import torch
    data = request.json
    if not data or 'speakers' not in data:
        return jsonify({'success': False, 'error': 'Invalid import data'}), 400
    
    imported = 0
    skipped = 0
    speaker_memory = get_speaker_memory()
    
    for name, info in data['speakers'].items():
        if name in speaker_memory:
            skipped += 1
            continue
        
        emb = torch.tensor(info['embedding']).float()
        speaker_memory[name] = emb
        imported += 1
    
    save_speaker_memory()
    
    return jsonify({
        'success': True,
        'imported': imported,
        'skipped': skipped,
        'total': len(speaker_memory)
    })


@api.route('/speakers/similarity-matrix', methods=['GET'])
def speaker_similarity_matrix():
    """Get similarity matrix between all enrolled speakers."""
    speaker_memory = get_speaker_memory()
    names = list(speaker_memory.keys())
    n = len(names)
    
    if n == 0:
        return jsonify({'success': True, 'matrix': [], 'names': []})
    
    # Convert all embeddings to numpy
    embeddings = []
    for name in names:
        emb = speaker_memory[name]
        if hasattr(emb, 'cpu'):
            emb_np = emb.cpu().numpy().flatten()
        else:
            emb_np = np.array(emb).flatten()
        embeddings.append(emb_np)
    
    # Calculate similarity matrix
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            sim = float(np.dot(embeddings[i], embeddings[j]) / 
                       (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8))
            row.append(round(sim, 3))
        matrix.append(row)
    
    return jsonify({
        'success': True,
        'names': names,
        'matrix': matrix
    })


@api.route('/clear', methods=['POST'])
def clear_speakers():
    """Clear all enrolled speakers."""
    clear_all_speakers()
    return jsonify({'success': True, 'message': 'All speakers cleared'})


@api.route('/reset-session-speakers', methods=['POST'])
def reset_session_speakers():
    """Reset session speaker clustering for a fresh start."""
    global session_audio_chunks, session_start_time
    reset_session()
    # Also clear session audio recording
    session_audio_chunks = []
    session_start_time = None
    return jsonify({'success': True})


# =============================================================================
# VibeVoice Post-Process Routes
# =============================================================================

def map_vibevoice_speakers_to_enrolled(segments, full_audio_path, sr=16000):
    """
    Map VibeVoice speaker numbers (0, 1, 2...) to enrolled speaker names.
    Uses ECAPA embeddings to find best match for each speaker.
    
    Returns dict: {0: "akash", 1: "#F31D04", ...}
    """
    if not segments:
        return {}
    
    speaker_memory = get_speaker_memory()
    speaker_mapping = {}
    
    # Load full audio
    try:
        full_wav, _ = librosa.load(full_audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading audio for speaker mapping: {e}")
        return {}
    
    # Group segments by speaker number
    speaker_segments = {}
    for seg in segments:
        spk_num = seg.get('Speaker', 0)
        if spk_num not in speaker_segments:
            speaker_segments[spk_num] = []
        speaker_segments[spk_num].append(seg)
    
    # For each unique speaker, extract representative audio and match
    for spk_num, segs in speaker_segments.items():
        # Get longest segment for this speaker (more reliable embedding)
        best_seg = max(segs, key=lambda s: s.get('End', 0) - s.get('Start', 0))
        start_sec = best_seg.get('Start', 0)
        end_sec = best_seg.get('End', 0)
        
        # Extract audio segment
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        
        if end_sample <= start_sample or end_sample > len(full_wav):
            # Fallback: use first 3 seconds of this speaker's audio
            start_sample = 0
            end_sample = min(3 * sr, len(full_wav))
        
        segment_wav = full_wav[start_sample:end_sample]
        
        if len(segment_wav) < sr * 0.5:  # Less than 0.5 seconds
            speaker_mapping[spk_num] = f"Speaker{spk_num}"
            continue
        
        try:
            # Get embedding for this speaker segment
            with processing_lock:
                segment_embedding = extract_embedding(segment_wav, sr)
            
            if hasattr(segment_embedding, 'cpu'):
                seg_emb_np = segment_embedding.cpu().numpy().flatten()
            else:
                seg_emb_np = np.array(segment_embedding).flatten()
            
            # Compare with enrolled speakers
            best_name = None
            best_sim = 0.0
            
            for name, enrolled_emb in speaker_memory.items():
                if hasattr(enrolled_emb, 'cpu'):
                    enrolled_np = enrolled_emb.cpu().numpy().flatten()
                else:
                    enrolled_np = np.array(enrolled_emb).flatten()
                
                sim = float(np.dot(seg_emb_np, enrolled_np) / 
                           (np.linalg.norm(seg_emb_np) * np.linalg.norm(enrolled_np) + 1e-8))
                
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            
            # Use enrolled name if similarity is high enough, otherwise use hash
            if best_name and best_sim >= 0.35:
                speaker_mapping[spk_num] = best_name
                print(f"Speaker{spk_num} -> {best_name} (sim={best_sim:.3f})")
            else:
                # Generate hash from embedding
                speaker_hash = embedding_to_hash(seg_emb_np, length=6)
                speaker_mapping[spk_num] = f"#{speaker_hash}"
                print(f"Speaker{spk_num} -> #{speaker_hash} (best_sim={best_sim:.3f})")
                
        except Exception as e:
            print(f"Error mapping speaker {spk_num}: {e}")
            speaker_mapping[spk_num] = f"Speaker{spk_num}"
    
    return speaker_mapping


@api.route('/post-process', methods=['POST'])
def post_process():
    """
    Run VibeVoice post-processing on recorded session audio.
    Combines all chunks, transcribes with diarization, maps speakers.
    """
    global session_audio_chunks
    
    if not session_audio_chunks:
        return jsonify({
            'success': False,
            'error': 'No audio recorded. Start playback first.'
        }), 400
    
    # Get settings from request
    data = request.get_json() or {}
    context = data.get('context', '')
    enable_sampling = data.get('enable_sampling', False)
    temperature = float(data.get('temperature', 0.0))
    top_p = float(data.get('top_p', 1.0))
    
    try:
        import time
        start_time = time.time()
        
        # Combine all audio chunks into single WAV
        all_audio = np.concatenate([chunk[0] for chunk in session_audio_chunks])
        total_duration = len(all_audio) / session_audio_sample_rate
        
        print(f"Post-process: {len(session_audio_chunks)} chunks, {total_duration:.2f}s total")
        
        # Save combined audio to temp file
        combined_path = os.path.join(UPLOAD_FOLDER, 'session_combined.wav')
        sf.write(combined_path, all_audio, session_audio_sample_rate)
        
        # Initialize VibeVoice if needed
        if not is_vibevoice_loaded():
            print("Initializing VibeVoice model...")
            if not init_vibevoice():
                return jsonify({
                    'success': False,
                    'error': 'Failed to initialize VibeVoice model'
                }), 500
        
        # Run VibeVoice transcription
        segments = transcribe_with_vibevoice(
            combined_path,
            context=context,
            enable_sampling=enable_sampling,
            temperature=temperature,
            top_p=top_p
        )
        
        if not segments:
            return jsonify({
                'success': False,
                'error': 'VibeVoice transcription returned no results'
            }), 500
        
        # Map speaker numbers to enrolled names
        speaker_mapping = map_vibevoice_speakers_to_enrolled(
            segments, 
            combined_path, 
            session_audio_sample_rate
        )
        
        # Apply speaker mapping to segments
        for seg in segments:
            spk_num = seg.get('Speaker', 0)
            seg['Speaker'] = speaker_mapping.get(spk_num, f"Speaker{spk_num}")
        
        elapsed = time.time() - start_time
        
        return jsonify({
            'success': True,
            'segments': segments,
            'stats': {
                'processing_time': round(elapsed, 2),
                'audio_duration': round(total_duration, 2),
                'chunk_count': len(session_audio_chunks),
                'speaker_mapping': speaker_mapping
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Post-process error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/post-process/status', methods=['GET'])
def post_process_status():
    """Get post-process session status."""
    vibevoice_status = get_vibevoice_status()
    
    chunk_count = len(session_audio_chunks)
    if chunk_count > 0:
        total_samples = sum(len(chunk[0]) for chunk in session_audio_chunks)
        duration = total_samples / session_audio_sample_rate
    else:
        duration = 0
    
    return jsonify({
        'success': True,
        'has_audio': chunk_count > 0,
        'chunk_count': chunk_count,
        'duration': round(duration, 2),
        'vibevoice': vibevoice_status
    })


@api.route('/init-vibevoice', methods=['POST'])
def init_vibevoice_route():
    """Initialize VibeVoice model."""
    if not VIBEVOICE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'VibeVoice not available - mlx-audio not installed'
        }), 400
    
    success = init_vibevoice()
    if success:
        return jsonify({
            'success': True,
            'message': 'VibeVoice model initialized'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to initialize VibeVoice model'
        }), 500


# =============================================================================
# Transcription Routes
# =============================================================================

@api.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    """Transcribe audio using the currently loaded ASR model (Parakeet or Voxtral)."""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    # Check if any model is loaded
    if not is_parakeet_loaded() and not is_voxtral_loaded():
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
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_transcribe.wav')
        
        wav, sr = load_audio_from_bytes(audio_data, audio_format, 16000, noise_level)
        sf.write(temp_path, wav, sr)
        
        # Transcribe with whichever model is loaded (Parakeet preferred for speed)
        text = transcribe_audio(temp_path)
        model_used = get_current_asr_model_name()
        
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


# =============================================================================
# Streaming Processing Route
# =============================================================================

@api.route('/process-streaming', methods=['POST'])
def process_streaming():
    """Process small audio chunk for streaming transcription with speaker detection."""
    if not is_model_loaded():
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
        
        wav, sr = load_audio_from_bytes(audio_data, audio_format, 16000, noise_level)
        
        # Check for valid audio
        valid, reason = is_audio_valid(wav, sr)
        if not valid:
            return jsonify({
                'success': True,
                'text': '',
                'speaker': 'Unknown',
                'is_sentence_end': False,
                'speaker_changed': False
            })
        
        # Record audio chunk for post-processing
        global session_audio_chunks, session_start_time
        import time
        if session_start_time is None:
            session_start_time = time.time()
        current_time = time.time() - session_start_time
        session_audio_chunks.append((wav.copy(), current_time))
        
        # Use lock for GPU operations to prevent race condition
        with processing_lock:
            # Save for transcription (fast, small chunk)
            temp_path = os.path.join(UPLOAD_FOLDER, 'temp_stream.wav')
            sf.write(temp_path, wav, sr)
            
            # Transcribe with whichever model is loaded
            text = transcribe_audio(temp_path)
            if not text:
                text = ''
            
            # Always extract per-chunk embedding for history tracking
            chunk_embedding = extract_embedding(wav, sr)
            
            # Accumulate audio for speaker ID (need longer chunks for stability)
            add_audio_chunk(wav)
            
            # Store in chunk history
            if hasattr(chunk_embedding, 'cpu'):
                chunk_emb_np = chunk_embedding.cpu().numpy().flatten()
            else:
                chunk_emb_np = np.array(chunk_embedding).flatten()
            
            chunk_idx = add_chunk_to_history(chunk_emb_np, text)
            
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
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            
            # Calculate total accumulated duration
            total_duration = get_accumulated_duration(sr)
            
            # Variables for response
            stable_embedding_ready = False
            speaker_change_detected = False
            split_at_index = None
            
            # If immediate speaker change detected, reset accumulation for faster ID
            if immediate_speaker_change:
                # Clear old accumulated audio - start fresh with new speaker
                set_accumulated_audio([wav])  # Start fresh with current chunk
                print("Accumulation reset due to immediate speaker change")
            
            # Only do stable speaker detection when we have enough audio
            if total_duration >= ACCUM_DURATION_FOR_SPEAKER_ID:
                # Concatenate accumulated audio
                accumulated_audio = get_accumulated_audio()
                accumulated_wav = np.concatenate(accumulated_audio)
                current_embedding = extract_embedding(accumulated_wav, sr)
                stable_embedding_ready = True
                
                # Keep last 2 seconds for continuity (sliding window)
                keep_samples = int(2.0 * sr)
                if len(accumulated_wav) > keep_samples:
                    set_accumulated_audio([accumulated_wav[-keep_samples:]])
                else:
                    set_accumulated_audio([accumulated_wav])
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
        speaker_memory = get_speaker_memory()
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
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        # Try hybrid detection on EVERY chunk (not just after 3 seconds)
        print(f"Hybrid check: pyannote={is_pyannote_loaded()}, prev_emb={prev_emb is not None}")
        if is_pyannote_loaded() or prev_emb is not None:
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
        last_stable_speaker = get_last_stable_speaker()
        if stable_embedding_ready:
            if last_stable_speaker is not None and speaker != last_stable_speaker:
                if not speaker_changed:  # Only if hybrid didn't already detect
                    speaker_changed = True
                    change_method = 'stable_3s'
                    print(f"Stable speaker change: {last_stable_speaker} -> {speaker}")
                
                # Find exact split point by checking chunk history
                chunk_history = get_chunk_history()
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
            set_last_stable_speaker(speaker)
        
        return jsonify({
            'success': True,
            'text': text,
            'speaker': speaker,
            'similarity': round(best_sim, 3),
            'is_sentence_end': is_sentence_end,
            'speaker_changed': speaker_changed,
            'change_method': change_method,
            'split_at_chunk_index': split_at_chunk_index,
            'chunk_index': get_chunk_index(),
            'embedding': current_emb_np.flatten().tolist()  # For next comparison
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Auto-Enroll Route
# =============================================================================

@api.route('/auto-enroll', methods=['POST'])
def auto_enroll():
    """
    Auto-enroll a speaker based on spoken name.
    Expects: audio file and transcript (from browser speech recognition or OpenAI)
    """
    if not is_model_loaded():
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
        
        wav, sr = load_audio_from_bytes(audio_data, 'webm', 16000, noise_level)
        
        # Check audio quality
        duration = len(wav) / sr
        if duration < 1.0:
            return jsonify({'success': False, 'error': f'Recording too short ({duration:.1f}s)'}), 400
        
        if np.max(np.abs(wav)) < 0.01:
            return jsonify({'success': False, 'error': 'No audio detected'}), 400
        
        # Extract embedding and store
        emb = extract_embedding(wav, sr)
        add_speaker(name, emb)
        
        speaker_memory = get_speaker_memory()
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
# Sample Audio Route
# =============================================================================

@api.route('/load-sample', methods=['GET'])
def load_sample():
    """Load sample.wav for testing purposes."""
    import numpy as np
    
    # Use sample.wav from static folder
    sample_path = os.path.join('static', 'sample.wav')
    
    if not os.path.exists(sample_path):
        return jsonify({'success': False, 'error': 'Sample file not found'}), 404
    
    try:
        # Load audio to get duration and waveform
        audio, sr = librosa.load(sample_path, sr=16000, mono=True)
        duration = len(audio) / sr
        
        # Generate waveform data for visualization
        samples_per_pixel = max(1, len(audio) // 1000)
        waveform_data = []
        for i in range(0, len(audio), samples_per_pixel):
            chunk = audio[i:i + samples_per_pixel]
            if len(chunk) > 0:
                waveform_data.append(float(np.max(np.abs(chunk))))
        
        # Normalize waveform
        max_val = max(waveform_data) if waveform_data else 1
        waveform_data = [v / max_val for v in waveform_data]
        
        return jsonify({
            'success': True,
            'audio_path': sample_path,
            'audio_url': '/static/sample.wav',
            'title': 'Sample Audio',
            'duration': duration,
            'waveform_data': waveform_data
        })
        
    except Exception as e:
        log.error(f"Error loading sample: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# YouTube Routes
# =============================================================================

@api.route('/youtube/download', methods=['POST'])
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
    output_path = os.path.join(UPLOAD_FOLDER, f'youtube_{video_id}.wav')
    
    try:
        import yt_dlp
        
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
            set_youtube_job(job_id, {
                'status': 'ready',
                'title': title,
                'duration': duration_str,
                'audio_path': output_path,
                'audio_url': audio_url
            })
            
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
        
        set_youtube_job(job_id, {
            'status': 'ready',
            'title': info['title'],
            'duration': info['duration_str'],
            'audio_path': output_path,
            'audio_url': audio_url
        })
        
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


@api.route('/youtube/process', methods=['POST'])
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
        output_path = os.path.join(UPLOAD_FOLDER, f'youtube_{job_id}.wav')
        info = download_youtube_audio(url, output_path)
        
        # Initialize job with audio URL
        audio_url = f'/api/youtube/audio/{job_id}'
        
        set_youtube_job(job_id, {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting...',
            'transcript': [],
            'speakers': [],
            'title': info['title'],
            'duration': info['duration_str'],
            'audio_path': output_path,
            'audio_url': audio_url
        })
        
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


@api.route('/youtube/audio/<job_id>', methods=['GET'])
def youtube_audio(job_id):
    """Serve YouTube audio file with CORS headers for Web Audio API."""
    job = get_youtube_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    audio_path = job.get('audio_path')
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    response = make_response(send_file(audio_path, mimetype='audio/wav'))
    # Add CORS headers for Web Audio API access
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@api.route('/youtube/status/<job_id>', methods=['GET'])
def youtube_status(job_id):
    """Get status of a YouTube processing job."""
    job = get_youtube_job(job_id)
    if not job:
        return jsonify({'status': 'error', 'error': 'Job not found'}), 404
    
    return jsonify(job)
