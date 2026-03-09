# -*- coding: utf-8 -*-
"""
Speaker management for Speaker Identification System.

Handles speaker memory, enrollment, identification, and session clustering.
"""

import os
import pickle
import numpy as np
import torch

import logger as log

from config import (
    SPEAKER_MEMORY_FILE,
    SPEAKER_CLUSTER_THRESHOLD,
    ADAPTIVE_UPDATE_THRESHOLD,
    ADAPTIVE_LEARNING_RATE,
    MAX_CHUNK_HISTORY,
    ACCUM_DURATION_FOR_SPEAKER_ID,
    MAX_SESSION_SPEAKERS
)
from utils import embedding_to_hash

# =============================================================================
# Global State
# =============================================================================

# Enrolled speakers (persistent)
speaker_memory = {}  # name -> embedding

# Session-based speaker clustering for stable IDs
session_speakers = {}  # hash -> averaged_embedding
session_speaker_counts = {}  # hash -> count (for averaging)

# Adaptive update tracking
adaptive_update_counter = {}  # Track updates per speaker

# Audio accumulation for stable speaker ID
accumulated_audio = []  # List of audio chunks

# Chunk history for retroactive split detection
chunk_history = []  # List of {embedding, text, index}
last_stable_speaker = None  # Last speaker from 3-second stable detection
chunk_index_counter = 0  # Global counter for chunk ordering

# =============================================================================
# Persistence Functions
# =============================================================================

def save_speaker_memory():
    """Save enrolled speakers to disk."""
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
        log.info(f"Saved {len(save_dict)} enrolled speakers to {SPEAKER_MEMORY_FILE}")
    except Exception as e:
        log.error(f"Error saving speaker memory: {e}")


def load_speaker_memory():
    """Load enrolled speakers from disk."""
    global speaker_memory
    try:
        if os.path.exists(SPEAKER_MEMORY_FILE):
            with open(SPEAKER_MEMORY_FILE, 'rb') as f:
                speaker_memory = pickle.load(f)
            log.info(f"Loaded {len(speaker_memory)} enrolled speakers from {SPEAKER_MEMORY_FILE}")
    except Exception as e:
        log.error(f"Error loading speaker memory: {e}")


# =============================================================================
# Speaker Management Functions
# =============================================================================

def get_speaker_memory():
    """Get the speaker memory dictionary."""
    return speaker_memory


def get_session_speakers():
    """Get session speakers dictionary."""
    return session_speakers


def get_session_speaker_counts():
    """Get session speaker counts dictionary."""
    return session_speaker_counts


def add_speaker(name, embedding):
    """Add or update a speaker in memory."""
    speaker_memory[name] = embedding
    save_speaker_memory()


def remove_speaker(name):
    """Remove a speaker from memory."""
    if name in speaker_memory:
        del speaker_memory[name]
        save_speaker_memory()
        return True
    return False


def rename_speaker(old_name, new_name):
    """Rename a speaker."""
    if old_name not in speaker_memory:
        return False, "Speaker not found"
    if new_name in speaker_memory and new_name != old_name:
        return False, f"Speaker '{new_name}' already exists"
    
    speaker_memory[new_name] = speaker_memory.pop(old_name)
    save_speaker_memory()
    return True, None


def clear_all_speakers():
    """Clear all enrolled speakers."""
    speaker_memory.clear()
    save_speaker_memory()


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
        log.info(f"Adaptive update: '{speaker_name}' profile saved (update #{count})")
    else:
        log.debug(f"Adaptive update: '{speaker_name}' profile updated (update #{count})")
    
    return True


# =============================================================================
# Session Speaker Clustering
# =============================================================================

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
        log.debug(f"Matched existing speaker {best_match} (sim={best_sim:.3f})")
        return best_match
    else:
        # New speaker - create new hash
        new_hash = embedding_to_hash(emb_np, length=6)
        session_speakers[new_hash] = emb_np.copy()
        session_speaker_counts[new_hash] = 1
        log.debug(f"New speaker detected: {new_hash} (best_sim={best_sim:.3f})")
        
        # Cleanup: remove oldest speakers if over limit
        if len(session_speakers) > MAX_SESSION_SPEAKERS:
            _cleanup_session_speakers()
        
        return new_hash


def _cleanup_session_speakers():
    """Remove oldest session speakers when over limit."""
    global session_speakers, session_speaker_counts
    
    # Sort by count (lowest = oldest/least seen)
    sorted_speakers = sorted(session_speaker_counts.items(), key=lambda x: x[1])
    
    # Remove oldest 10 speakers
    remove_count = min(10, len(sorted_speakers) - MAX_SESSION_SPEAKERS + 10)
    for speaker_hash, _ in sorted_speakers[:remove_count]:
        del session_speakers[speaker_hash]
        del session_speaker_counts[speaker_hash]
        log.debug(f"Session cleanup: removed speaker {speaker_hash}")
    
    log.info(f"Session speakers cleaned: {len(session_speakers)} remaining")


def reset_session():
    """Reset session speaker clustering for a fresh start."""
    global session_speakers, session_speaker_counts, accumulated_audio, chunk_history, last_stable_speaker, chunk_index_counter
    session_speakers = {}
    session_speaker_counts = {}
    accumulated_audio = []
    chunk_history = []
    last_stable_speaker = None
    chunk_index_counter = 0
    log.info("Session speakers, audio buffer, and chunk history reset")


def add_to_session_speakers(name, embedding):
    """Add an enrolled speaker to session speakers for matching."""
    if hasattr(embedding, 'cpu'):
        session_emb = embedding.cpu().numpy().flatten()
    else:
        session_emb = np.array(embedding).flatten()
    session_speakers[name] = session_emb
    session_speaker_counts[name] = 1


# =============================================================================
# Audio Accumulation
# =============================================================================

def get_accumulated_audio():
    """Get accumulated audio chunks."""
    return accumulated_audio


def add_audio_chunk(wav):
    """Add audio chunk to accumulation."""
    accumulated_audio.append(wav)


def set_accumulated_audio(chunks):
    """Set accumulated audio chunks."""
    global accumulated_audio
    accumulated_audio = chunks


def get_accumulated_duration(sr=16000):
    """Get total duration of accumulated audio."""
    total_samples = sum(len(chunk) for chunk in accumulated_audio)
    return total_samples / sr


# =============================================================================
# Chunk History
# =============================================================================

def get_chunk_history():
    """Get chunk history."""
    return chunk_history


def add_chunk_to_history(embedding, text, index=None):
    """Add a chunk to history."""
    global chunk_index_counter
    if index is None:
        chunk_index_counter += 1
        index = chunk_index_counter
    
    if hasattr(embedding, 'cpu'):
        emb_np = embedding.cpu().numpy().flatten()
    else:
        emb_np = np.array(embedding).flatten()
    
    chunk_history.append({
        'embedding': emb_np.copy(),
        'text': text,
        'index': index
    })
    
    # Keep only last N chunks
    if len(chunk_history) > MAX_CHUNK_HISTORY:
        chunk_history.pop(0)
    
    return index


def get_chunk_index():
    """Get current chunk index counter."""
    return chunk_index_counter


def get_last_stable_speaker():
    """Get last stable speaker."""
    return last_stable_speaker


def set_last_stable_speaker(speaker):
    """Set last stable speaker."""
    global last_stable_speaker
    last_stable_speaker = speaker


# =============================================================================
# Speaker Identification
# =============================================================================

def identify_speaker(embedding, threshold=0.5):
    """Identify speaker from embedding.
    
    Args:
        embedding: speaker embedding
        threshold: similarity threshold
        
    Returns:
        tuple: (identified_name, best_score, all_scores)
    """
    if hasattr(embedding, 'cpu'):
        emb_np = embedding.cpu().numpy().flatten()
    else:
        emb_np = np.array(embedding).flatten()
    
    best_name = None
    best_score = -1.0
    all_scores = {}
    
    for name, ref_emb in speaker_memory.items():
        if hasattr(ref_emb, 'cpu'):
            ref_np = ref_emb.cpu().numpy().flatten()
        else:
            ref_np = np.array(ref_emb).flatten()
        
        # Cosine similarity
        score = float(np.dot(emb_np, ref_np) / 
                     (np.linalg.norm(emb_np) * np.linalg.norm(ref_np) + 1e-8))
        all_scores[name] = round(score, 4)
        
        if score > best_score:
            best_score = score
            best_name = name
    
    # Return identified name only if above threshold
    if best_name and best_score >= threshold:
        return best_name, best_score, all_scores
    else:
        return None, best_score, all_scores


def enroll_from_session_hash(speaker_hash, name, merge_with_existing=False):
    """Enroll a speaker from their session hash.
    
    Args:
        speaker_hash: session speaker hash (without #)
        name: name to assign
        merge_with_existing: if True, merge with existing speaker embedding
        
    Returns:
        tuple: (success, message, update_count)
    """
    global speaker_memory, adaptive_update_counter
    
    # Remove # prefix if present
    if speaker_hash.startswith('#'):
        speaker_hash = speaker_hash[1:]
    
    speaker_hash = speaker_hash.upper()
    
    if speaker_hash not in session_speakers:
        return False, f"Unknown speaker hash: {speaker_hash}. Speaker may have expired from session.", 0
    
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
            log.debug(f"Adaptive merge: '{name}' learned from hash #{speaker_hash}, update #{update_count}")
        else:
            # Create new speaker or overwrite
            speaker_memory[name] = emb_tensor
            message = f"Speaker '{name}' enrolled from fingerprint"
            update_count = 0
        
        # Save to disk
        save_speaker_memory()
        
        return True, message, update_count
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, str(e), 0


# Load saved speakers on module import
load_speaker_memory()
