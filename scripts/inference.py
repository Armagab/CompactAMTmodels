import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pretty_midi

import time
import torch
from thop import profile
from thop import clever_format


def get_piano_roll(midi_path, sr, hop_length):
    """Extract Ground Truth Piano Roll from MIDI."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    roll = pm.get_piano_roll(fs=sr/hop_length)
    roll = roll[21:109, :] 
    return (roll > 0).astype(np.float32)


def predict(model, audio, n_bins, sr, hop, device):
    
    if n_bins == 229:
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=hop, n_mels=n_bins)
        S_db = librosa.power_to_db(S, ref=np.max)
    else:
        C = librosa.cqt(y=audio, sr=sr, hop_length=hop, n_bins=n_bins, bins_per_octave=36)
        S_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    

    inp = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple):
            logits = out[1]
        else:
            logits = out
            
        probs = torch.sigmoid(logits)
        

    return probs.squeeze(0).T.cpu().numpy(), S_db


def plot_comparison(spectrogram, pred_roll, gt_roll, sr, hop_length, threshold):

    min_len = min(spectrogram.shape[1], pred_roll.shape[1], gt_roll.shape[1], 2000)
    spectrogram = spectrogram[:, :min_len]
    pred_roll = pred_roll[:, :min_len]
    gt_roll = gt_roll[:, :min_len]

    times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop_length)
    note_range = (21, 108)

    plt.figure(figsize=(12, 10))

    # 1. Spectrogram
    plt.subplot(3, 1, 1)
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(label='dB')
    plt.title("Audio Spectrogram")
    plt.xlabel('')

    # 2. Predicted Piano Roll
    plt.subplot(3, 1, 2)
    
    pred_binary = (pred_roll > threshold).astype(float)
    
    plt.imshow(pred_binary, aspect='auto', origin='lower',
               extent=[times[0], times[-1], note_range[0], note_range[1]],
               cmap='Blues', interpolation='nearest')
    plt.ylabel('MIDI Note')
    plt.title(f"Predicted MIDI")
    plt.xlabel('')

    # 3. Ground Truth Piano Roll
    plt.subplot(3, 1, 3)
    plt.imshow(gt_roll, aspect='auto', origin='lower',
               extent=[times[0], times[-1], note_range[0], note_range[1]],
               cmap='Greens', interpolation='nearest')
    plt.ylabel('MIDI Note')
    plt.xlabel('Time (s)')
    plt.title("Ground Truth MIDI")

    plt.tight_layout()
    plt.show()


def measure_efficiency(model, device='cpu', duration_sec=60, sr=22050, hop=512, n_bins=229):
    """
    Measures model parameters, FLOPS и inference time for one minute of audio.
    """
    model.to(device)
    model.eval()
    
    
    frames_count = int(duration_sec * sr / hop)
    dummy_input = torch.randn(1, 1, n_bins, frames_count).to(device)
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # FLOPS (Macs)
    try:
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")
    except Exception as e:
        print(f"FLOPS calculation failed: {e}")
        macs, macs_str = 0, "N/A"

    # Latency
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    # Timing everything
    start_time = time.time()
    n_runs = 10
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
            
    avg_time = (time.time() - start_time) / n_runs
    rtf = avg_time / duration_sec # Real-Time Factor
    
    print(f"--- Efficiency Report ({device.upper()}) ---")
    print(f"Model: {type(model).__name__}")
    print(f"Input duration: {duration_sec} sec")
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"MACs (FLOPs/2): {macs_str}")
    print(f"Avg Inference Time: {avg_time:.4f} sec")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    print(f"Speedup vs Real-time: {1/rtf:.1f}x")
    
    return {
        "params": total_params,
        "macs": macs,
        "inference_time": avg_time,
        "rtf": rtf
    }
