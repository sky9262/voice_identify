# -*- coding: utf-8 -*-
"""
Speaker Identification Web Server

A Flask web application for speaker verification using Microsoft's WavLM model.
"""

import os
import io
import re
import torch
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from pydub import AudioSegment

# Optional: OpenAI for better transcription
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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

MODEL_ID = "microsoft/wavlm-base-plus-sv"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables (initialized later)
feature_extractor = None
spk_model = None
speaker_memory = {}  # name -> embedding
model_loaded = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_model():
    """Initialize the speaker embedding model."""
    global feature_extractor, spk_model, model_loaded
    
    if model_loaded:
        return True
    
    print(f"Loading model on device: {device}")
    print(f"Model: {MODEL_ID}")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    spk_model = AutoModelForAudioXVector.from_pretrained(MODEL_ID).to(device)
    spk_model.eval()
    
    model_loaded = True
    print("Model loaded successfully!")
    return True


@torch.no_grad()
def extract_embedding(wav, sr=16000):
    """Extract speaker embedding from audio waveform."""
    if not model_loaded:
        raise RuntimeError("Model not initialized")
    
    inputs = feature_extractor(wav, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = spk_model(**inputs)
    
    if hasattr(outputs, "embeddings"):
        emb = outputs.embeddings
    else:
        emb = outputs.last_hidden_state.mean(dim=1)
    
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.squeeze(0)


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
    return jsonify({
        'model_loaded': model_loaded,
        'device': device,
        'speakers_enrolled': list(speaker_memory.keys()),
        'speaker_count': len(speaker_memory)
    })


@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize the model."""
    try:
        init_model()
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
        
        # Load and process audio
        wav, sr = load_audio(filepath, target_sr=16000)
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
        
        # Load and process audio
        wav, sr = load_audio(filepath, target_sr=16000)
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


@app.route('/api/clear', methods=['POST'])
def clear_speakers():
    """Clear all enrolled speakers."""
    speaker_memory.clear()
    return jsonify({'success': True, 'message': 'All speakers cleared'})


@app.route('/api/identify-live', methods=['POST'])
def identify_live():
    """Identify speaker from live audio chunk (WebM/Opus from browser)."""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if not speaker_memory:
        return jsonify({'success': False, 'error': 'No speakers enrolled'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    threshold = float(request.form.get('threshold', 0.5))
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Convert WebM/Opus to WAV using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        # Export to WAV bytes
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Load with librosa
        wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        
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
            if score > best_score:
                best_score = score
                best_name = name
        
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
            'all_scores': all_scores
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/enroll-live', methods=['POST'])
def enroll_live():
    """Enroll a speaker from live audio (WebM/Opus from browser)."""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Speaker name is required'}), 400
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Convert WebM/Opus to WAV using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        # Export to WAV bytes
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Load with librosa
        wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        
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
    """Transcribe audio using OpenAI Whisper (if available) or return error."""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio data'}), 400
    
    use_openai = request.form.get('use_openai', 'false').lower() == 'true'
    api_key = request.form.get('api_key', '').strip()
    
    if use_openai:
        if not OPENAI_AVAILABLE:
            return jsonify({'success': False, 'error': 'OpenAI package not installed. Run: pip install openai'}), 400
        if not api_key:
            return jsonify({'success': False, 'error': 'OpenAI API key required'}), 400
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Convert WebM to WAV
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        if use_openai and api_key:
            # Use OpenAI Whisper
            client = openai.OpenAI(api_key=api_key)
            wav_buffer.name = "audio.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer
            )
            text = transcript.text
        else:
            # Return that browser speech recognition should be used
            return jsonify({
                'success': True,
                'use_browser': True,
                'message': 'Use browser speech recognition'
            })
        
        return jsonify({
            'success': True,
            'text': text,
            'use_browser': False
        })
    except Exception as e:
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
    
    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Convert WebM/Opus to WAV
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        wav, sr = librosa.load(wav_buffer, sr=16000, mono=True)
        
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
# Main
# =============================================================================

if __name__ == '__main__':
    print("Starting Speaker Identification Web Server...")
    print(f"Device: {device}")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
