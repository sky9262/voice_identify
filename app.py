# -*- coding: utf-8 -*-
"""
Speaker Identification Web Server

Main entry point - Using SpeechBrain ECAPA-TDNN for high-accuracy speaker verification.

This file has been refactored into modules:
- config.py: Configuration and constants
- utils.py: Shared utility functions  
- audio.py: Audio processing functions
- models.py: Model loading and inference
- speakers.py: Speaker management
- youtube.py: YouTube processing
- routes.py: API endpoints
"""

from flask import Flask, render_template

from config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH, device
from routes import api

# =============================================================================
# Flask App Setup
# =============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Register API blueprint
app.register_blueprint(api, url_prefix='/api')


# =============================================================================
# Main Route
# =============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("Starting Speaker Identification Web Server...")
    print(f"Device: {device}")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
