import os
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch                  
from torch.utils.data import Dataset, DataLoader


class PianoRollDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        split: str,
        spec_type: str,
        chunk_len: int = 1024,
        random_crop: bool = True
    ):
        super().__init__() 

        self.root_dir = Path(root_dir)                      
        self.split = split                                   
        self.spec_type = spec_type                              
        self.chunk_len = chunk_len                                 
        self.random_crop = random_crop                               


        split_dir = self.root_dir / f"{split}_split"                


        if spec_type == "mel":
            self.audio_dir = split_dir / "audio_mel"
        elif spec_type == "cqt":
            self.audio_dir = split_dir / "audio_cqt"
        else:
            raise ValueError("spec_type has to be 'mel' or 'cqt'")


        self.midi_dir = split_dir / "midi"


        audio_files = sorted([f for f in os.listdir(self.audio_dir) if f.endswith(".npy")])
        midi_files = sorted([f for f in os.listdir(self.midi_dir) if f.endswith(".npy")])


        audio_map = {Path(f).stem: self.audio_dir / f for f in audio_files}
        midi_map = {Path(f).stem: self.midi_dir / f for f in midi_files}


        common_keys = sorted(set(audio_map.keys()) & set(midi_map.keys()))

        self.pairs: List[Tuple[Path, Path]] = []
        for key in common_keys:
            self.pairs.append((audio_map[key], midi_map[key]))


    def __len__(self):
        return len(self.pairs)

    def _load_pair(self, idx: int):
        audio_path, midi_path = self.pairs[idx]

        spec = np.load(audio_path).astype(np.float32)
        pianoroll = np.load(midi_path).astype(np.float32)




        if pianoroll.shape[0] == 88 and pianoroll.shape[1] == spec.shape[1]:
            pianoroll = pianoroll.T

        elif pianoroll.shape[1] == 88:
            pass 
        else:
            raise ValueError(f"Wrong pianoroll dimensions: {pianoroll.shape}.")

        return spec, pianoroll

    def _crop_or_pad(self, spec: np.ndarray, pianoroll: np.ndarray):

        F, T = spec.shape

        if self.chunk_len is None:
            return spec, pianoroll

        if T > self.chunk_len:
            # выбираем начало кропа
            if self.random_crop:
                start = np.random.randint(0, T - self.chunk_len + 1)
            else:
                start = 0
            end = start + self.chunk_len

            spec = spec[:, start:end]
            pianoroll = pianoroll[start:end, :]
        elif T < self.chunk_len:
            pad = self.chunk_len - T

            spec = np.pad(spec, ((0, 0), (0, pad)), mode="constant")
            pianoroll = np.pad(pianoroll, ((0, pad), (0, 0)), mode="constant")

        return spec, pianoroll

    def __getitem__(self, idx: int):
        spec, pianoroll = self._load_pair(idx)
        spec, pianoroll = self._crop_or_pad(spec, pianoroll)

        pianoroll_mask = (pianoroll > 0).astype(np.float32)

        spec_tensor = torch.from_numpy(spec).unsqueeze(0)
        target_tensor = torch.from_numpy(pianoroll_mask)

        return spec_tensor, target_tensor
        

def create_dataloaders(
    root_dir: str,
    spec_type: str,
    batch_size: int = 8,
    chunk_len: int = 1024,
    num_workers: int = 4
):


    train_ds = PianoRollDataset(
        root_dir=root_dir,
        split="train",
        spec_type=spec_type,
        chunk_len=chunk_len,
        random_crop=True
    )

    val_ds = PianoRollDataset(
        root_dir=root_dir,
        split="val",
        spec_type=spec_type,
        chunk_len=chunk_len,
        random_crop=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader