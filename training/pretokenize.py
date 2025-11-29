"""
Script to pre-tokenize training/validation data for Sesame finetuning and save incrementally in HDF5.

Usage:
python pretokenize.py --train_data /path/to/train.json --val_data /path/to/val.json --output /path/to/output/data.hdf5
"""

import argparse
from pathlib import Path
import sqlite3
import pandas as pd
import torch
import torchaudio
import h5py
import numpy as np
from tqdm import tqdm

from utils import load_tokenizers, MIMI_SAMPLE_RATE, AUDIO_NUM_CODEBOOKS


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="./data/tokens.hdf5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=100, help="Save every N samples")
    parser.add_argument("--omit_speaker_id", action="store_true", help="Don't prepend text with a speaker id")
    args = parser.parse_args(arg_string.split() if arg_string else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def load_metadata(data_path: Path | str) -> pd.DataFrame:
    """
    Load metadata from various formats.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)

    if data_path.suffix == ".json":
        return pd.read_json(data_path)
    elif data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    elif data_path.suffix == ".sql":
        return pd.read_sql_query("SELECT * FROM data", sqlite3.connect(data_path))
    elif data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    elif data_path.suffix == ".pkl":
        return pd.read_pickle(data_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {data_path}")


def append_to_hdf5(file_path, split, audio_tokens_batch, text_tokens_batch, compression="gzip"):
    """
    Append audio, text, and length information to the HDF5 file.
    Audio is flattened (vlen) for space efficiency.
    """
    with h5py.File(file_path, "a") as f:
        grp = f.require_group(split)

        vlen_dtype = h5py.special_dtype(vlen=np.int32)
        audio_ds = grp.get("audio") or grp.create_dataset("audio", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        text_ds = grp.get("text") or grp.create_dataset("text", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        length_ds = grp.get("length") or grp.create_dataset("length", shape=(0,), maxshape=(None,), dtype=np.int32)

        n = len(audio_tokens_batch)
        audio_ds.resize(audio_ds.shape[0] + n, axis=0)
        text_ds.resize(text_ds.shape[0] + n, axis=0)
        length_ds.resize(length_ds.shape[0] + n, axis=0)

        for i in range(n):
            audio_array = np.array(audio_tokens_batch[i], dtype=np.int32).flatten()  # [n_codebooks * seq_len]
            text_array = np.array(text_tokens_batch[i], dtype=np.int32)

            seq_len = audio_array.shape[0] // AUDIO_NUM_CODEBOOKS
            total_len = seq_len + len(text_array) + 1  # +1 for EOS frame

            audio_ds[-n + i] = audio_array
            text_ds[-n + i] = text_array
            length_ds[-n + i] = total_len


def get_num_existing_samples(file_path, split):
    """Return the number of existing samples in the HDF5 file for the given split, using the 'length' dataset."""
    try:
        with h5py.File(file_path, "r") as f:
            return f[split]["length"].shape[0]
    except Exception:
        return 0


def tokenize_and_store(data_path, output_path, split, audio_tokenizer, text_tokenizer, device, save_every=100):
    """
    Tokenize the dataset and save in HDF5 incrementally, resuming if interrupted.
    """
    df = load_metadata(data_path)
    n_existing = get_num_existing_samples(output_path, split)
    if n_existing:
        print(f"⏩ Resuming {split}: skipping {n_existing} already processed samples")
        df = df.iloc[n_existing:]
    else:
        print(f"🔄 Processing {split} split: {len(df)} samples")

    audio_tokens_batch, text_tokens_batch = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Handle optional timestamps
        if "start" in row and "end" in row:
            sr = torchaudio.info(row["path"]).sample_rate
            frame_offset = int(row["start"] * sr)
            num_frames = int((row["end"] - row["start"]) * sr)
        else:
            frame_offset = 0
            num_frames = -1

        # Load and resample audio
        waveform, sr = torchaudio.load(row["path"], frame_offset=frame_offset, num_frames=num_frames)
        waveform = torchaudio.functional.resample(waveform.squeeze(0), orig_freq=sr, new_freq=MIMI_SAMPLE_RATE)
        waveform = waveform.unsqueeze(0).unsqueeze(0).to(device)

        # Tokenize
        audio_tokens = audio_tokenizer.encode(waveform)[0].tolist()  # shape: [n_codebooks, seq_len]
        speaker = row.get("speaker", 999)
        text = f"[{speaker}]{row['text']}" if not args.omit_speaker_id else row['text']
        text_tokens = text_tokenizer.encode(text)

        # Accumulate batch
        audio_tokens_batch.append(audio_tokens)
        text_tokens_batch.append(text_tokens)

        if len(audio_tokens_batch) >= save_every:
            append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch)
            audio_tokens_batch, text_tokens_batch = [], []

    # Final flush
    if audio_tokens_batch:
        append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    tokenize_and_store(
        args.train_data, output_path=args.output, split="train",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, save_every=args.save_every
    )

    tokenize_and_store(
        args.val_data, output_path=args.output, split="val",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, save_every=args.save_every
    )

    print(f"\n✅ Done. Tokenized data saved to: {args.output}")
