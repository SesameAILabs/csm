import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import h5py

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
AUDIO_NUM_CODEBOOKS = int(os.getenv("AUDIO_NUM_CODEBOOKS"))


class TokenizedDataset(Dataset):
    """
    HDF5-backed dataset for tokenized audio and text samples.

    Assumes audio is saved as flat vlen int32 arrays (flattened [n_codebooks, seq_len]).
    Optionally loads the entire split into memory for faster access.
    """
    def __init__(self, token_dataset_path: str, split: str, load_in_memory: bool = False):
        assert token_dataset_path.endswith(".hdf5"), "Token dataset path must end with .hdf5"
        self.token_dataset_path = token_dataset_path
        self.split = split
        self._file = None  # Lazy open in __getitem__
        self._in_memory = load_in_memory
        self._audio = None
        self._text = None

        # Read length once (for __len__)
        with h5py.File(token_dataset_path, "r") as f:
            self._length = len(f[f"{split}/audio"])
            if self._in_memory:
                self._audio = [torch.tensor(a, dtype=torch.long) for a in f[f"{split}/audio"][:]]
                self._text = [torch.tensor(t, dtype=torch.long) for t in f[f"{split}/text"][:]]

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        if self._in_memory:
            flat_audio = self._audio[idx]
            text = self._text[idx]
        else:
            if self._file is None:
                self._file = h5py.File(self.token_dataset_path, "r")
            flat_audio = torch.tensor(self._file[f"{self.split}/audio"][idx], dtype=torch.long)
            text = torch.tensor(self._file[f"{self.split}/text"][idx], dtype=torch.long)

        audio = flat_audio.view(AUDIO_NUM_CODEBOOKS, -1)
        return {"audio": audio, "text": text}


def collate_fn(batch: List[dict]):
    """
    Collate function for tokenized audio and text.
    Merges variable-length audio/text into a single padded tensor.
    """
    tokens, tokens_mask = [], []

    for item in batch:
        audio_tokens = item["audio"]  # [n_codebooks, audio_seq_len]
        text_tokens = item["text"]    # [text_seq_len]

        # Add EOS frame to audio
        eos_frame = torch.zeros(audio_tokens.size(0), 1)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        # extra dimension is for text tokens
        audio_frame = torch.zeros(audio_tokens.size(1), AUDIO_NUM_CODEBOOKS + 1).long()
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), AUDIO_NUM_CODEBOOKS + 1).bool()
        audio_frame_mask[:, :-1] = True

        # Format text frame with same shape
        text_frame = torch.zeros(len(text_tokens), AUDIO_NUM_CODEBOOKS + 1).long()
        text_frame[:, -1] = text_tokens
        text_frame_mask = torch.zeros(len(text_tokens), AUDIO_NUM_CODEBOOKS + 1).bool()
        text_frame_mask[:, -1] = True

        # Concatenate and collect
        tokens.append(torch.cat([text_frame, audio_frame], dim=0))
        tokens_mask.append(torch.cat([text_frame_mask, audio_frame_mask], dim=0))

    tokens = pad_sequence(tokens, batch_first=True)
    tokens_mask = pad_sequence(tokens_mask, batch_first=True, padding_value=False)

    return tokens, tokens_mask


class BucketSampler(Sampler):
    """
    Groups samples of similar lengths into bins to minimize padding.
    """
    def __init__(
        self, lengths: List[int], batch_size: int, shuffle: bool = True,
        is_infinite: bool = True, random_seed: int = 42
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.is_infinite = is_infinite
        self.random_seed = random_seed
        self.local_step = 0
        self.bins = self._create_bins(lengths, batch_size)

    def _create_bins(self, lengths: List[int], batch_size: int) -> List[List[int]]:
        indices_with_lengths = sorted(enumerate(lengths), key=lambda x: x[1])
        bins, current_bin = [], []

        for idx, _ in indices_with_lengths:
            current_bin.append(idx)
            if len(current_bin) >= batch_size:
                bins.append(current_bin)
                current_bin = []

        if current_bin:
            bins.append(current_bin)

        return bins

    def _shuffle_bins(self, epoch: int):
        rng = np.random.RandomState(epoch + self.random_seed)
        rng.shuffle(self.bins)
        for bin_ in self.bins:
            rng.shuffle(bin_)

    def __iter__(self):
        epoch = 0
        while True:
            if self.shuffle:
                self._shuffle_bins(epoch)
            for bin_indices in self.bins:
                yield bin_indices
                self.local_step += 1
            if not self.is_infinite:
                break
            epoch += 1

    def __len__(self):
        return len(self.bins)


def load_lengths(token_dataset_path: str, split: str) -> List[int]:
    with h5py.File(token_dataset_path, "r") as f:
        return list(f[f"{split}/length"][:])


def create_dataloaders(
    token_dataset_path: str,
    batch_size: int,
    infinite_train: bool = False,
    load_in_memory: bool = False,
    num_workers: int = 0,
):
    """
    Creates training and validation dataloaders from an HDF5 file.
    Optionally loads the entire dataset into memory for efficiency.
    """
    train_lengths = load_lengths(token_dataset_path, "train")
    val_lengths = load_lengths(token_dataset_path, "val")

    trainset = TokenizedDataset(token_dataset_path, split="train", load_in_memory=load_in_memory)
    valset = TokenizedDataset(token_dataset_path, split="val", load_in_memory=load_in_memory)

    trainsampler = BucketSampler(
        lengths=train_lengths, batch_size=batch_size,
        is_infinite=infinite_train, shuffle=True
    )

    valsampler = BucketSampler(
        lengths=val_lengths, batch_size=batch_size,
        is_infinite=False, shuffle=False
    )

    trainloader = DataLoader(
        trainset, batch_sampler=trainsampler,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    valloader = DataLoader(
        valset, batch_sampler=valsampler,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    return trainloader, valloader
