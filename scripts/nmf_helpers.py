import os
import re

import librosa

import numpy as np
from scipy.ndimage import median_filter


def activations_to_notes(H, frame_times, pitches_midi,
                                 onset_rel_thresh=0.3,
                                 min_frames=2):
    notes = []
    R, T = H.shape

    for i in range(R):
        act = H[i]
        if act.max() <= 0:
            continue

        thr = onset_rel_thresh * act.max()
        active = act >= thr

        t = 0
        while t < T:
            if active[t]:
                start = t
                while t < T and active[t]:
                    t += 1
                end = t
                if end - start >= min_frames:
                    onset = frame_times[start]
                    offset = frame_times[end] if end < T else frame_times[-1]
                    pitch_midi = pitches_midi[i]
                    notes.append((onset, offset, pitch_midi))
            t += 1
    return notes

def build_w_fixed(isolated_dir, sr, n_fft, hop_length):

    note_names = []
    basis_vectors = []

    for filename in sorted(os.listdir(isolated_dir)):
        if not filename.lower().endswith(".wav"):
            continue

        note_name = os.path.splitext(filename)[0]
        y, _ = librosa.load(os.path.join(isolated_dir, filename), sr=sr, mono=True)


        y, _ = librosa.effects.trim(y, top_db=40)

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S = np.abs(D)


        basis = S.mean(axis=1)

        
        if basis.max() > 0:
            basis = basis / (basis.max() + 1e-8)

        basis_vectors.append(basis)
        note_names.append(note_name)

    W_fixed = np.column_stack(basis_vectors).astype(np.float32)
    return W_fixed, note_names


def denoise_activations(H,
                        rel_thresh=0.2,
                        top_k=5,
                        smooth_win=3,
                        min_frames=2):
    R, T = H.shape
    Hc = H.copy()


    for t in range(T):
        col = Hc[:, t]
        m = col.max()
        if m <= 0:
            Hc[:, t] = 0
            continue

        col[col < m * rel_thresh] = 0

        if top_k is not None and top_k > 0:
            idx = np.argsort(col)[::-1]
            keep = idx[:top_k]
            mask = np.zeros_like(col, dtype=bool)
            mask[keep] = True
            col[~mask] = 0

        Hc[:, t] = col


    if smooth_win and smooth_win > 1:
        for r in range(R):
            if Hc[r].max() > 0:
                Hc[r] = median_filter(Hc[r], size=smooth_win)


    for r in range(R):
        row = Hc[r]
        active = row > 0
        i = 0
        while i < T:
            if active[i]:
                start = i
                while i < T and active[i]:
                    i += 1
                end = i
                if (end - start) < min_frames:
                    row[start:end] = 0
            i += 1
        Hc[r] = row

    return Hc

def note_name_to_midi(note_name):
    note_to_num = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4,
                   'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    match = re.match(r'^([A-G]#?)(\d)$', note_name)
    if not match:
        raise ValueError(f"Invalid note name: {note_name}")
    note, octave = match.groups()
    return 12 + int(octave) * 12 + note_to_num[note]