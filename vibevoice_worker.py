#!/usr/bin/env python3
"""
VibeVoice-ASR worker - runs transcription in a separate process.
This isolates the mlx native code so segfaults don't crash the main server.
"""
import sys
import json
import os
import time
import traceback


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: vibevoice_worker.py <audio_path> <output_path> [context]"}))
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2]
    context = sys.argv[3] if len(sys.argv) > 3 else ""
    
    if not os.path.exists(audio_path):
        print(json.dumps({"error": f"Audio file not found: {audio_path}"}))
        sys.exit(1)
    
    try:
        start_time = time.time()
        
        # Load model
        from mlx_audio.stt.utils import load_model
        from mlx_audio.stt.generate import generate_transcription
        
        model_id = "mlx-community/VibeVoice-ASR-4bit"
        print(f"Loading VibeVoice model: {model_id}", file=sys.stderr)
        model = load_model(model_id)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s", file=sys.stderr)
        
        # Run transcription
        json_output_path = os.path.splitext(output_path)[0] + "_vibevoice.json"
        
        print(f"Transcribing: {audio_path}", file=sys.stderr)
        result = generate_transcription(
            model,
            audio_path,
            json_output_path,
            "json",
            True,
            text=context if context else ""
        )
        
        elapsed = time.time() - start_time
        print(f"Transcription completed in {elapsed:.2f}s", file=sys.stderr)
        
        # Parse results
        segments = []
        
        # Check result object
        if hasattr(result, 'segments') and result.segments:
            for seg in result.segments:
                segments.append({
                    'Start': seg.get('start', 0),
                    'End': seg.get('end', 0),
                    'Speaker': seg.get('speaker_id', seg.get('speaker', 0)),  # VibeVoice uses speaker_id
                    'Content': seg.get('text', '')
                })
        
        # Check output JSON file
        if not segments and os.path.exists(json_output_path):
            try:
                with open(json_output_path, 'r') as f:
                    data = json.load(f)
                raw_segments = []
                if isinstance(data, list):
                    raw_segments = data
                elif isinstance(data, dict) and 'segments' in data:
                    raw_segments = data['segments']
                
                # Normalize segment keys
                for seg in raw_segments:
                    segments.append({
                        'Start': seg.get('start', seg.get('Start', 0)),
                        'End': seg.get('end', seg.get('End', 0)),
                        'Speaker': seg.get('speaker_id', seg.get('speaker', seg.get('Speaker', 0))),
                        'Content': seg.get('text', seg.get('Content', ''))
                    })
            except Exception:
                pass
        
        # Fallback: single segment from text
        if not segments and hasattr(result, 'text') and result.text:
            segments = [{
                'Start': 0,
                'End': 0,
                'Speaker': 0,
                'Content': result.text
            }]
        
        # Output result as JSON to stdout
        output = {
            "success": True,
            "segments": segments,
            "elapsed": round(elapsed, 2)
        }
        print(json.dumps(output))
        
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
