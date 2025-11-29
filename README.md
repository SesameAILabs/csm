# CSM

**2025/05/20** - CSM is availabile natively in [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/csm) 🤗 as of version `4.52.1`, more info available [in our model repo](https://huggingface.co/sesame/csm-1b)

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Quickstart

This script will generate a conversation between 2 characters, using a prompt for each character.

```bash
python run_csm.py
```

## Usage

If you want to write your own applications with CSM, the following examples show basic usage.

#### Generate a sentence

This will use a random speaker identity, as no prompt or context is provided.

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

#### Generate with context

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

NOTE: The following example is instructional and the audio files do not exist. It is intended as an example for using context with CSM.

```python
from generator import Segment

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## Fine-tuning / Forking this Project

The official repository primarily contains inference code. **This fork includes a `training/` folder** with scripts for fine-tuning CSM on new languages like Norwegian, Swedish, and Danish.

See [`training/README.md`](training/README.md) for full documentation.

### Quick Overview

```bash
# 1. Pre-tokenize your audio data
python training/pretokenize.py --train_data data/train.json --val_data data/val.json

# 2. Train (full fine-tuning recommended for new languages)
python training/train.py --data data/tokens.hdf5 --output_dir experiments/scandinavian
```

### Training Scripts Source

Scripts adapted from community forks:

- [knottwill/sesame-finetune](https://github.com/knottwill/sesame-finetune) - Full fine-tuning (recommended for new languages)
- [davidbrowne17/csm-streaming](https://github.com/davidbrowne17/csm-streaming) - LoRA fine-tuning (for voice cloning)
- [rooms-solutions/csm-multilingual](https://github.com/rooms-solutions/csm-multilingual) - Multilingual training effort

### Preparing Audio Data with Mimi

CSM uses [Mimi](https://huggingface.co/kyutai/mimi) (from Kyutai) as its audio codec. Before training, you need to convert your audio files to Mimi tokens. Here's how:

```python
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders

# Load Mimi
device = "cuda" if torch.cuda.is_available() else "cpu"
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(32)

def audio_to_mimi_tokens(audio_path: str) -> torch.Tensor:
    """
    Convert an audio file to Mimi tokens for training.

    Args:
        audio_path: Path to audio file (WAV recommended)

    Returns:
        Tensor of shape (num_codebooks, num_frames) containing audio tokens
    """
    # Load and resample audio to Mimi's sample rate (24kHz)
    audio, sample_rate = torchaudio.load(audio_path)

    # Ensure single channel
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != mimi.sample_rate:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sample_rate, new_freq=mimi.sample_rate
        )

    # Encode to tokens: (1, 1, num_samples) -> (1, num_codebooks, num_frames)
    audio = audio.squeeze(0).to(device)
    tokens = mimi.encode(audio.unsqueeze(0).unsqueeze(0))

    return tokens[0]  # (num_codebooks, num_frames)

# Example usage
tokens = audio_to_mimi_tokens("your_audio.wav")
print(f"Token shape: {tokens.shape}")
```

### Non-English Fine-tuning Notes

- **Tokenizer**: CSM uses the Llama-3.2 tokenizer which handles most Latin alphabets (including Norwegian æ, ø, å) reasonably well.
- **Data quality**: For conversational models, include both high-quality recorded speech (for voice quality) and natural conversation data (for prosody and timing).
- **Hardware**: Fine-tuning typically requires a GPU with at least 24GB VRAM (e.g., RTX 4090, A100).

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.
