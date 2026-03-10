# Voice Identify

Real-time speaker identification and transcription system with voice enrollment capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS%20Optimized-orange)

## Features

### Core Features
- **Real-time Speaker Identification** - Identify who is speaking using WavLM embeddings
- **Voice Enrollment** - Register speakers by name for automatic recognition
- **Live Transcription** - Real-time speech-to-text with speaker labels
- **YouTube Audio Processing** - Download and process YouTube videos for speaker analysis
- **Live Microphone Mode** - Record and identify speakers in real-time
- **Sentence-level Segmentation** - Split transcripts at natural sentence boundaries
- **Speaker Change Detection** - Hybrid detection using Pyannote + embedding similarity
- **Retroactive Speaker Labeling** - Update past transcript entries when speakers are enrolled

### Audio Visualization
- **8 Waveform Styles** - Smooth Wave, Frequency Bars, Circular, Mirror, Oscilloscope, Particles, 3D Terrain, Ribbon
- **Real-time Controls** - Adjustable smoothing, points, fade trail, line width, amplitude, glow
- **Persistent Settings** - Save/Reset waveform preferences (per-tab)
- **Available in both YouTube and Microphone modes**

### Speaker Management
- **Speaker Timeline** - Visual timeline showing when each speaker talked
- **Export/Import Speakers** - Backup and restore enrolled speaker profiles
- **Speaker Similarity Matrix** - View embedding similarity between speakers
- **Auto-Fix Speakers** - Automatic correction of unidentified speaker labels
- **Click-to-Merge** - Manually merge hash-labeled speakers with enrolled names

### Transcript Features
- **Clickable Timestamps** - Click any line to seek audio playback
- **Transcript Search** - Find specific words/phrases with highlighting
- **Typing Effect** - Smooth character-by-character text rendering
- **VibeVoice Post-Processing** - Enhanced transcription with progress indicator

## Tech Stack

| Component | Technology |
|-----------|------------|
| Speaker Embeddings | Microsoft WavLM (wavlm-base-plus-sv) |
| ASR (Speech-to-Text) | Parakeet-MLX / Voxtral-MLX |
| Speaker Segmentation | Pyannote Audio |
| Backend | Flask (Python) |
| Audio Processing | librosa, pydub, noisereduce |
| GPU Acceleration | Apple MPS (with CPU fallback) |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sky9262/voice_identify.git
cd voice_identify
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set HuggingFace token for Pyannote

Create a `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

Required for Pyannote speaker segmentation model access.

## Usage

### Start the server

```bash
python app.py
```

Server runs at: `http://localhost:5001`

### First-time Setup

1. Open the web interface
2. Go to **Settings** tab
3. Click **Initialize All Models** (downloads ~81MB ECAPA-TDNN model)
4. Select and initialize your preferred ASR model

### Speaker Enrollment

1. Go to **Manage** tab
2. Upload a clear audio sample of the speaker
3. Enter the speaker's name
4. Click **Enroll Speaker**

### Processing Audio

**YouTube Tab:**
- Paste a YouTube URL
- Click **Download Audio**
- Click **Start Live Processing**
- Play the audio to see real-time transcription with speaker labels

**Live Microphone:**
- Switch to **Live Microphone** mode
- Click **Start Recording**
- Speak into your microphone
- See real-time speaker identification

### Sample Audio

A 30-minute sample audio file is included at `static/sample.wav` for testing.

## Project Structure

```
voice_identify/
├── app.py              # Flask application entry point
├── routes.py           # API endpoints
├── models.py           # ML model loading and inference
├── speakers.py         # Speaker memory and clustering
├── audio.py            # Audio processing utilities
├── youtube.py          # YouTube download handling
├── config.py           # Configuration constants
├── utils.py            # Shared utilities
├── logger.py           # Logging configuration
├── templates/
│   └── index.html      # Web interface
├── static/
│   └── sample.wav      # Sample audio for testing
├── uploads/            # Temporary audio files
└── pretrained_models/  # Auto-downloaded models
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/init` | POST | Initialize speaker ID model |
| `/api/status` | GET | Get system status |
| `/api/enroll` | POST | Enroll a new speaker |
| `/api/speakers` | GET | List enrolled speakers |
| `/api/speakers/<name>` | DELETE | Remove a speaker |
| `/api/process-streaming` | POST | Process audio chunk |
| `/api/youtube/download` | POST | Download YouTube audio |
| `/api/switch-model` | POST | Switch ASR model |

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SPEAKER_CLUSTER_THRESHOLD` | 0.15 | Similarity threshold for speaker matching |
| `ECAPA_SIMILARITY_THRESHOLD` | 0.30 | Threshold for speaker change detection |
| `MAX_SESSION_SPEAKERS` | 50 | Maximum speakers per session |
| `ACCUM_DURATION_FOR_SPEAKER_ID` | 3.0 | Seconds to accumulate before speaker ID |

## Requirements

- Python 3.9+
- macOS (Apple Silicon recommended) or Linux
- ~2GB disk space for models
- Microphone (for live mode)

## License

MIT License

## Author

[sky9262](https://github.com/sky9262)
