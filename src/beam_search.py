"""
Beam Search decoding for CRNN CTC output
"""

import numpy as np
from collections import defaultdict

BLANK = 0

def simple_beam_search(probs, beam_width=5):
    """
    Simple beam search for CTC decoding

    Args:
        probs: [T, C] probability array
        beam_width: number of beams to keep

    Returns:
        Best sequence (list of class indices)
    """
    T, C = probs.shape
    beams = [(tuple(), 0.0)]  # (sequence, log_prob)

    for t in range(T):
        new_beams = defaultdict(lambda: -1e9)

        for seq, score in beams:
            for c in range(C):
                new_seq = seq + (c,)
                new_score = score + float(np.log(max(probs[t, c], 1e-12)))
                if new_score > new_beams[new_seq]:
                    new_beams[new_seq] = new_score

        # Keep top beam_width beams
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = beams[0][0]

    # Collapse repeats & remove blank
    out = []
    prev = -1
    for c in best_seq:
        if c != prev and c != BLANK:
            out.append(c)
        prev = c

    return out

def beam_search(probs, beam_width=5):
    """
    Main beam search function compatible with app.py

    Args:
        probs: [T, C] probability array or torch tensor
        beam_width: number of beams to keep

    Returns:
        List of decoded strings (only first one is used)
    """
    # Convert torch tensor to numpy if needed
    if hasattr(probs, 'cpu'):
        probs = probs.cpu().numpy()

    if len(probs.shape) == 3:  # (B, T, C) -> (T, C)
        probs = probs[0]

    # Get best sequence
    seq = simple_beam_search(probs, beam_width)

    # Convert to string representation (this will be handled by caller)
    # For now, return as list with single element
    return [seq]

def beam_search_with_language_model(probs, vocab, language_model=None, beam_width=5):
    """
    Beam search with language model rescoring

    Args:
        probs: [T, C] probability array
        vocab: vocabulary list
        language_model: optional language model for rescoring
        beam_width: number of beams to keep

    Returns:
        Best decoded text
    """
    T, C = probs.shape

    # First pass: get beam search sequences
    raw_seq = simple_beam_search(probs, beam_width)

    # Convert to text
    text = "".join([vocab[i] for i in raw_seq if i < len(vocab)])

    # If no language model, return raw text
    if language_model is None:
        return text

    # Apply language model correction
    corrected_text = language_model.correct_text(text)
    return corrected_text

def greedy_decode(probs):
    """
    Greedy decoding for CTC output (simplest method)

    Args:
        probs: [T, C] probability array

    Returns:
        Decoded sequence
    """
    seq = probs.argmax(axis=1).tolist()

    # Collapse repeats & remove blank
    result = []
    prev = -1
    for idx in seq:
        if idx != prev and idx != BLANK:
            result.append(idx)
        prev = idx

    return result
