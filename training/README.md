# CSM Training Module

Training scripts for fine-tuning CSM on new languages (Norwegian, Swedish, Danish, etc.)

## Important: Full Fine-tuning vs LoRA for New Languages

> **For domain shifts like new languages, full fine-tuning is recommended over LoRA.**
> LoRA works well for voice adaptation, but modifying the original weights directly
> is optimal when teaching the model a new language.
> — [HuggingFace Discussion](https://huggingface.co/sesame/csm-1b/discussions/9)

## Files

| File | Source | Description |
|------|--------|-------------|
| `train.py` | knottwill/sesame-finetune | Full training loop with WandB logging |
| `pretokenize.py` | knottwill/sesame-finetune | Pre-tokenize audio/text to HDF5 |
| `dataloaders.py` | knottwill/sesame-finetune | Efficient data loading |
| `utils.py` | knottwill/sesame-finetune | Model loading, tokenizers, utilities |
| `lora.py` | davidbrowne17/csm-streaming | LoRA fine-tuning (for voice cloning) |
| `configs/` | knottwill/sesame-finetune | Training configurations |

## Quick Start

### 1. Prepare your data

Create a JSON file with your audio data:
```json
[
  {"audio_path": "path/to/audio1.wav", "text": "Hei, hvordan har du det?", "speaker_id": 0},
  {"audio_path": "path/to/audio2.wav", "text": "Jeg har det bra, takk!", "speaker_id": 1}
]
```

### 2. Pre-tokenize

```bash
python training/pretokenize.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --output data/tokens.hdf5
```

### 3. Train

**Full fine-tuning** (requires ~40GB+ VRAM):
```bash
python training/train.py \
  --data data/tokens.hdf5 \
  --output_dir experiments/norwegian \
  --config training/configs/finetune_param_defaults.yaml
```

**LoRA fine-tuning** (works on 24GB VRAM):
```bash
python training/lora.py
```

## Hardware Requirements

| Task | VRAM | Notes |
|------|------|-------|
| Inference | 8GB+ | RTX 3070 or better |
| LoRA fine-tuning | 24GB | RTX 4090, A10 |
| Full fine-tuning | 40GB+ | A100, H100 |

Mixed precision training (`--use_amp`) reduces memory usage significantly.

## Scandinavian Language Data Sources

- **NB Tale** (Norwegian): https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-31/
- **NPSC** (Norwegian Parliament): https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-58/
- **NST Swedish**: https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-56/
- **NST Danish**: https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-55/
- **Mozilla Common Voice**: https://commonvoice.mozilla.org/ (Norwegian, Swedish, Danish available)

## Tips for Scandinavian Languages

1. **Full fine-tuning for new languages**: LoRA is great for voice cloning but for teaching a new language, modify the original weights
2. **Mix data sources**: Use both high-quality recordings (NB Tale) and conversational data (NPSC) for natural prosody
3. **Tokenizer**: Llama-3.2 handles æ, ø, å, ä, ö well out of the box
4. **Speaker IDs**: Use consistent speaker IDs per voice/person in your dataset
5. **Text normalization**: Consider adding language-specific preprocessing (numbers, abbreviations)

## Other Multilingual Resources

- [rooms-solutions/csm-multilingual](https://github.com/rooms-solutions/csm-multilingual) - Community multilingual training effort
- [Speechmatics Fine-tuning Guide](https://blog.speechmatics.com/sesame-finetune) - Detailed walkthrough

## Known Limitations

- The base CSM-1B model has "some capacity for non-English languages due to data contamination" but performs poorly
- Sesame plans to expand to 20+ languages in future releases
- The 1B model is relatively small; expect better results with more training data
