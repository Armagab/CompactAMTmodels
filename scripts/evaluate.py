import numpy as np
import mir_eval
import pretty_midi

import torch
from torch import nn
from typing import List, Tuple
import torch.backends.cudnn as cudnn

def evaluate_transcription_full(pred_midi, gt_midi, onset_tolerance=0.05, fs=100):

    results = {}

    pred_notes = [n for inst in pred_midi.instruments for n in inst.notes]
    gt_notes = [n for inst in gt_midi.instruments for n in inst.notes]


    if not gt_notes:
        zero_metrics = {"P": 0.0, "R": 0.0, "F1": 0.0}
        return {
            "Frame": zero_metrics,
            "Note_Onset": zero_metrics
        }
    
    if not pred_notes:
        zero_metrics = {"P": 0.0, "R": 0.0, "F1": 0.0}
        return {
            "Frame": zero_metrics,
            "Note_Onset": zero_metrics
        }

    ref_intervals = np.array([[n.start, n.end] for n in gt_notes])
    ref_pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in gt_notes])
    est_intervals = np.array([[n.start, n.end] for n in pred_notes])
    est_pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in pred_notes])

    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=None 
    )
    results['Note_Onset'] = {'P': p, 'R': r, 'F1': f1}

    
    ref_roll = (gt_midi.get_piano_roll(fs=fs) > 0).astype(int)
    est_roll = (pred_midi.get_piano_roll(fs=fs) > 0).astype(int)
    
    T_ref = ref_roll.shape[1]
    T_est = est_roll.shape[1]
    
    if T_ref != T_est:
        max_len = max(T_ref, T_est)
        if T_ref < max_len:
            ref_roll = np.pad(ref_roll, ((0, 0), (0, max_len - T_ref)))
        if T_est < max_len:
            est_roll = np.pad(est_roll, ((0, 0), (0, max_len - T_est)))


    tp = np.sum(ref_roll & est_roll)

    fp = np.sum(est_roll) - tp

    fn = np.sum(ref_roll) - tp
    
    eps = 1e-8
    
    frame_precision = tp / (tp + fp + eps)
    frame_recall = tp / (tp + fn + eps)
    frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall + eps)

    results['Frame'] = {'P': frame_precision, 'R': frame_recall, 'F1': frame_f1}

    return results

def evaluate_transcription_midi(pred_midi, gt_midi, onset_tolerance=0.05, pitch_tolerance=50.0):

    pred_notes = [n for inst in pred_midi.instruments for n in inst.notes]
    gt_notes = [n for inst in gt_midi.instruments for n in inst.notes]

    if not pred_notes and not gt_notes:
        return 1.0, 1.0, 1.0
    if not gt_notes:
        return 0.0, 0.0, 0.0
    if not pred_notes:
        return 0.0, 0.0, 0.0


    ref_intervals = np.array([[n.start, n.end] for n in gt_notes])
    ref_pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in gt_notes])
    est_intervals = np.array([[n.start, n.end] for n in pred_notes])
    est_pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in pred_notes])


    mir_eval.transcription.validate(ref_intervals, ref_pitches, est_intervals, est_pitches)
    P, R, F1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=None
    )
    return P, R, F1

def frame_level_f1(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:

    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        preds = probs >= threshold

        y_true = targets.bool()
        y_pred = preds.bool()

        tp = (y_true & y_pred).sum().item()
        fp = (~y_true & y_pred).sum().item()
        fn = (y_true & ~y_pred).sum().item()

        if tp == 0 and fp == 0 and fn == 0:
            return 1.0

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1


def extract_notes_from_roll(
    roll_bool,
    frame_duration: float
) -> List[Tuple[float, float, int]]:

    N, P = roll_bool.shape
    events: List[Tuple[float, float, int]] = []

    for pitch in range(P):
        col = roll_bool[:, pitch]
        active = False
        onset_frame = 0

        for t in range(N):
            if not active and col[t]:

                active = True
                onset_frame = t
            elif active and not col[t]:

                offset_frame = t
                onset_time = onset_frame * frame_duration
                offset_time = offset_frame * frame_duration
                events.append((onset_time, offset_time, pitch))
                active = False

        if active:
            offset_frame = N
            onset_time = onset_frame * frame_duration
            offset_time = offset_frame * frame_duration
            events.append((onset_time, offset_time, pitch))

    return events




def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
    hop_length: int = 512,
    sr: int = 22050,
    onset_tolerance: float = 0.05,
    compute_note_f1: bool = True,
):

    model.eval()

    total_loss = 0.0
    total_frame_f1 = 0.0
    num_batches = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for spec, target in dataloader:
            
            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(spec)

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1, target.shape[-1])
            )

            batch_f1 = frame_level_f1(logits, target, threshold=threshold)

            total_loss += loss.item()
            total_frame_f1 += batch_f1
            num_batches += 1
            if compute_note_f1:
                all_logits.append(logits.cpu())
                all_targets.append(target.cpu())

    avg_loss = total_loss / max(1, num_batches)
    avg_frame_f1 = total_frame_f1 / max(1, num_batches)

    return avg_loss, avg_frame_f1, None

def compute_frame_micro_metrics(
    model: nn.Module,
    dataloader,
    device: torch.device,
    threshold: float = 0.5,
):

    model.eval()
    tp = fp = tn = fn = 0
    eps = 1e-8

    with torch.no_grad():
        for spec, target in dataloader:

            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            out = model(spec)

            if isinstance(out, (tuple, list)) and len(out) >= 2:
                _, logits = out
            else:
                logits = out

            probs = torch.sigmoid(logits)
            preds = probs >= threshold

            y_true = target.bool()
            y_pred = preds.bool()

            tp += (y_true & y_pred).sum().item()
            fp += (~y_true & y_pred).sum().item()
            tn += (~y_true & ~y_pred).sum().item()
            fn += (y_true & ~y_pred).sum().item()

    if tp == 0 and fp == 0 and tn == 0 and fn == 0:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "frame_f1": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "frame_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_note_micro_metrics(
    model: nn.Module,
    dataloader,
    device: torch.device,
    threshold: float = 0.5,
    hop_length: int = 512,
    sr: int = 22050,
    onset_tolerance: float = 0.05,
    use_onsets_if_available: bool = True,
):

    model.eval()
    tp = fp = tn = fn = 0
    eps = 1e-8

    frame_duration = hop_length / float(sr)

    with torch.no_grad():
        for spec, target in dataloader:

            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            out = model(spec)

            onset_logits = None
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                onset_logits, frame_logits = out
            else:
                frame_logits = out

            frame_probs = torch.sigmoid(frame_logits).cpu().numpy()
            tgt = target.cpu().numpy()

            if onset_logits is not None and use_onsets_if_available:
                onset_probs = torch.sigmoid(onset_logits).cpu().numpy()
            else:
                onset_probs = None

            B, T, P = frame_probs.shape

            for b in range(B):
                frame_b = frame_probs[b]
                tgt_bin = tgt[b] >= 0.5

                if onset_probs is not None and use_onsets_if_available:
                    onset_b = onset_probs[b]
                    pred_bin = np.zeros_like(frame_b, dtype=bool)

                    for p in range(P):
                        active = False
                        for t in range(T):
                            f_on = frame_b[t, p] >= threshold
                            o_on = onset_b[t, p] >= threshold

                            if not active:
                                if f_on and o_on:
                                    active = True
                                    pred_bin[t, p] = True
                                else:
                                    pred_bin[t, p] = False
                            else:
                                if f_on:
                                    pred_bin[t, p] = True
                                else:
                                    active = False
                                    pred_bin[t, p] = False
                else:
                    pred_bin = frame_b >= threshold

                
                pred_events = extract_notes_from_roll(pred_bin, frame_duration)
                ref_events = extract_notes_from_roll(tgt_bin, frame_duration)

                if len(ref_events) == 0 and len(pred_events) == 0:
                    continue

                used_pred = [False] * len(pred_events)
                tp_local = 0
                tol = onset_tolerance

                for onset_ref, _, pitch_ref in ref_events:
                    best_j = None
                    best_diff = None

                    for j, (onset_pred, _, pitch_pred) in enumerate(pred_events):
                        if used_pred[j]:
                            continue
                        if pitch_pred != pitch_ref:
                            continue

                        diff = abs(onset_pred - onset_ref)
                        if diff <= tol:
                            if best_diff is None or diff < best_diff:
                                best_diff = diff
                                best_j = j

                    if best_j is not None:
                        used_pred[best_j] = True
                        tp_local += 1

                fp_local = len(pred_events) - tp_local
                fn_local = len(ref_events) - tp_local

                tp += tp_local
                fp += fp_local
                fn += fn_local

    if tp == 0 and fp == 0 and fn == 0:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "note_f1": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    accuracy = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "note_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def build_onset_targets_from_frames(frame_targets: torch.Tensor) -> torch.Tensor:

    onset_targets = torch.zeros_like(frame_targets)


    onset_targets[:, 0, :] = frame_targets[:, 0, :]


    diff = frame_targets[:, 1:, :] - frame_targets[:, :-1, :]
    onset_targets[:, 1:, :] = torch.clamp(diff, min=0.0)

    return onset_targets


def evaluate_crnn(
    model: nn.Module,
    dataloader,
    onset_criterion,
    frame_criterion,
    device: torch.device,
    threshold: float = 0.5,
    hop_length: int = 512,
    sr: int = 22050,
    onset_tolerance: float = 0.05,
    compute_note_f1: bool = False,
    onset_loss_weight: float = 1.0,
    frame_loss_weight: float = 1.0,
):

    model.eval()

    total_loss = 0.0
    total_onset_loss = 0.0
    total_frame_loss = 0.0
    total_frame_f1 = 0.0
    num_batches = 0


    note_tp = note_fp = note_fn = 0
    frame_duration = hop_length / float(sr)
    eps = 1e-8

    with torch.no_grad():
        for spec, target in dataloader:

            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            onset_logits, frame_logits = model(spec)

            B, T, P = frame_logits.shape

            onset_targets = build_onset_targets_from_frames(target)

            loss_onset = onset_criterion(
                onset_logits.reshape(B * T, P),
                onset_targets.reshape(B * T, P),
            )
            loss_frame = frame_criterion(
                frame_logits.reshape(B * T, P),
                target.reshape(B * T, P),
            )

            loss = onset_loss_weight * loss_onset + frame_loss_weight * loss_frame

            batch_f1 = frame_level_f1(frame_logits, target, threshold=threshold)

            total_loss += loss.item()
            total_onset_loss += loss_onset.item()
            total_frame_loss += loss_frame.item()
            total_frame_f1 += batch_f1
            num_batches += 1

            if compute_note_f1:
                probs = torch.sigmoid(frame_logits).cpu().numpy()
                tgt = target.cpu().numpy()

                B_, T_, P_ = probs.shape
                assert B_ == B and T_ == T and P_ == P

                for b in range(B):
                    pred_bin = probs[b] >= threshold
                    tgt_bin = tgt[b] >= 0.5

                    pred_events = extract_notes_from_roll(pred_bin, frame_duration)
                    ref_events = extract_notes_from_roll(tgt_bin, frame_duration)

                    if len(ref_events) == 0 and len(pred_events) == 0:
                        continue

                    used_pred = [False] * len(pred_events)
                    tp_local = 0
                    tol = onset_tolerance

                    for onset_ref, _, pitch_ref in ref_events:
                        best_j = None
                        best_diff = None

                        for j, (onset_pred, _, pitch_pred) in enumerate(pred_events):
                            if used_pred[j]:
                                continue
                            if pitch_pred != pitch_ref:
                                continue

                            diff_t = abs(onset_pred - onset_ref)
                            if diff_t <= tol:
                                if best_diff is None or diff_t < best_diff:
                                    best_diff = diff_t
                                    best_j = j

                        if best_j is not None:
                            used_pred[best_j] = True
                            tp_local += 1

                    fp_local = len(pred_events) - tp_local
                    fn_local = len(ref_events) - tp_local

                    note_tp += tp_local
                    note_fp += fp_local
                    note_fn += fn_local

    avg_total_loss = total_loss / max(1, num_batches)
    avg_onset_loss = total_onset_loss / max(1, num_batches)
    avg_frame_loss = total_frame_loss / max(1, num_batches)
    avg_frame_f1 = total_frame_f1 / max(1, num_batches)

    note_f1 = None
    if compute_note_f1:
        if note_tp == 0 and note_fp == 0 and note_fn == 0:
            note_f1 = 1.0
        else:
            precision = note_tp / (note_tp + note_fp + eps)
            recall = note_tp / (note_tp + note_fn + eps)
            note_f1 = 2 * precision * recall / (precision + recall + eps)

    return avg_total_loss, avg_frame_f1, avg_onset_loss, avg_frame_loss, note_f1