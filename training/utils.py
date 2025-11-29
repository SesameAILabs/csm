import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from pathlib import Path
from typing import Union
from torch.optim.lr_scheduler import LambdaLR
from torch import nn

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
CSM_REPO_PATH = os.getenv("CSM_REPO_PATH")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if CSM_REPO_PATH:
    sys.path.append(CSM_REPO_PATH)
    from generator import Generator, load_llama3_tokenizer, load_watermarker
    from models import Model, ModelArgs, _create_causal_mask
else:
    raise ValueError("CSM_REPO_PATH not set in .env file")

MIMI_SAMPLE_RATE = int(os.getenv("MIMI_SAMPLE_RATE", 24_000))
BACKBONE_FLAVOR = os.getenv("BACKBONE_FLAVOR", "llama-1B")
DECODER_FLAVOR = os.getenv("DECODER_FLAVOR", "llama-100M")
TEXT_VOCAB_SIZE = int(os.getenv("TEXT_VOCAB_SIZE", 128256))
AUDIO_VOCAB_SIZE = int(os.getenv("AUDIO_VOCAB_SIZE", 2051))
AUDIO_NUM_CODEBOOKS = int(os.getenv("AUDIO_NUM_CODEBOOKS", 32))


class WarmupDecayLR(LambdaLR):
    """
    Learning rate scheduler with a linear warmup and specificable decay.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, decay_type: str = "linear"):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_type = decay_type
        super().__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        else:
            if self.decay_type == "linear":
                return (self.total_steps - step) / (self.total_steps - self.warmup_steps)
            elif self.decay_type == "constant":
                return 1.0
            elif self.decay_type == "exponential":
                return 0.1 ** ((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            elif self.decay_type == "cosine":
                return 0.5 * (1 + torch.cos(torch.pi * torch.tensor((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))))
            else:
                raise ValueError(f"Invalid decay type: {self.decay_type}")
            

def load_tokenizers(device: Union[str, torch.device]):
    """Load text and audio tokenizers."""
    text_tokenizer = load_llama3_tokenizer()
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(AUDIO_NUM_CODEBOOKS)
    audio_tokenizer = mimi
    
    return text_tokenizer, audio_tokenizer


def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor):
    """
    Forward pass for Sesame's CSM model.
    This will be added to the model with `model.forward = types.MethodType(forward, model)`

    Args:
        tokens: (batch_size, seq_len, n_codebooks+1)
        tokens_mask: (batch_size, seq_len, n_codebooks+1)
    """
    dtype = next(self.parameters()).dtype
    bsz, seq_len, _ = tokens.size()
    device = tokens.device

    # embed tokens
    embeds = self._embed_tokens(tokens)

    # get targets and codebook embeddings corresponding to audio tokens
    audio_mask = tokens_mask[:, :, 0]  # [bsz, seq_len]
    target_tokens = tokens[audio_mask][:, :-1]  # [audio_len, n_codebooks]
    c_embeds = embeds[:, :, :-1, :][audio_mask]  # [audio_len, n_codebooks, embed_dim] 

    # retain just non-padding embeddings
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)

    # backbone forward pass
    padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]  # [bsz, seq_len]
    backbone_attn_mask = _create_causal_mask(seq_len, device)  # [seq_len, seq_len]
    padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)  # [bsz, seq_len, seq_len]
    backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d
    backbone_attn_mask = backbone_attn_mask | torch.eye(seq_len, device=device).bool().unsqueeze(0).expand(bsz, -1, -1)
    input_pos = torch.arange(0, seq_len).unsqueeze(0).expand(bsz, seq_len).long().to(device)
    h = self.backbone(h, input_pos=input_pos, mask=backbone_attn_mask).to(dtype=dtype)

    # get backbone embeddings used for audio codebook prediction
    audio_mask = torch.roll(audio_mask, -1, 1)  # shift audio mask to the right by 1
    audio_h = h[audio_mask]  # [audio_len, embed_dim]

    # predict first codebook and compute loss
    c0_logits = self.codebook0_head(audio_h)  # [audio_len, audio_vocab_size]
    c0_target = target_tokens[:, 0]  # [audio_len]
    c0_loss = F.cross_entropy(c0_logits, c0_target)

    # "compute amortization" (train decoder on random 1/16 subset of audio tokens)
    indices = torch.randperm(c_embeds.size(0))[: c_embeds.size(0) // 16]
    c_embeds = c_embeds[indices][:, :-1, :]  # [audio_len//16, n_codebooks-1, embed_dim]
    audio_h = audio_h[indices]  # [audio_len//16, embed_dim]
    target_tokens = target_tokens[indices][:, 1:]  # [audio_len//16, n_codebooks-1]

    # concatenate backbone embeddings and codebook embeddings for decoder input
    decoder_embeds = torch.cat(
        [audio_h.unsqueeze(1), c_embeds], dim=1
    )  # [audio_len//16, n_codebooks, embed_dim]
    N, n_codebooks, _ = decoder_embeds.size()
    c_pos = torch.arange(0, n_codebooks).unsqueeze(0).expand(N, n_codebooks).long().to(device)

    decoder_causal_mask = _create_causal_mask(decoder_embeds.size(1), device).expand(N, -1, -1)
    decoder_h = self.decoder(self.projection(decoder_embeds), input_pos=c_pos, mask=decoder_causal_mask).to(dtype=dtype)
    c_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)

    c_loss = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), target_tokens.reshape(-1))

    loss = 2 * ((1 - self.decoder_loss_weight) * c0_loss + self.decoder_loss_weight * c_loss)
    return loss


def init_weights(model: nn.Module):
    """
    Initialize the weights of the model.
    - Xavier uniform initialization for linear layers
    - Normal initialization for embeddings
    - Xavier uniform initialization for parameters
    """

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.data)

    model.apply(_init_weights)

    # Special handling for audio_head because it's nn.Parameter directly
    nn.init.xavier_uniform_(model.audio_head)

    return model

def load_model(
        model_name_or_checkpoint_path: Union[str, Path] = None,
        device: Union[str, torch.device] = 'cuda',
        decoder_loss_weight: float = 0.5
    ) -> Model:
    """Load model, add forward method, and move to device.
    
    Args:
        model_name_or_checkpoint_path: Name or path of pretrained model or checkpoint.
        device: Device to move the model to.
        decoder_loss_weight: Decoder loss weight.
    """
    if model_name_or_checkpoint_path is None or model_name_or_checkpoint_path.endswith(".pt"):
        # Training from scratch or using local checkpoint
        config = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=TEXT_VOCAB_SIZE,
            audio_vocab_size=AUDIO_VOCAB_SIZE,
            audio_num_codebooks=AUDIO_NUM_CODEBOOKS
        )
        model = Model(config)

        if model_name_or_checkpoint_path:
            state_dict = torch.load(model_name_or_checkpoint_path)['model']
            model.load_state_dict(state_dict)
        else:
            model = init_weights(model)
    else: 
        # Huggingface model name or local path
        model = Model.from_pretrained(model_name_or_checkpoint_path)

    model.decoder_loss_weight = decoder_loss_weight
    model.forward = types.MethodType(forward, model)  # add the forward method to the model

    model = model.to(device=device)
    return model


def reset_caches(model: Model):
    """Reset the caches of the model (used after each generation)."""
    model.reset_caches()
    for module in model.modules():
        if hasattr(module, "cache_enabled"):
            module.cache_enabled = False
        if hasattr(module, "kv_cache"):
            module.kv_cache = None


def custom_generator_init(self, model: Model, audio_tokenizer: torch.nn.Module, text_tokenizer, watermarker):
    """Custom __init__ for the Generator class (from sesame csm repo)."""
    self._model = model
    self._model.setup_caches(1)

    self._text_tokenizer = text_tokenizer

    device = next(model.parameters()).device
    self._audio_tokenizer = audio_tokenizer.to(device=device)
    self.sample_rate = MIMI_SAMPLE_RATE
    self.device = device

    self._watermarker = watermarker


def generate_audio(model, audio_tokenizer, text_tokenizer, watermarker, text, speaker_id, device, use_amp=True, max_audio_length_ms=10_000):
    """Generate audio from text."""
    model.eval()
    Generator.__init__ = types.MethodType(custom_generator_init, Generator)
    generator = Generator(model, audio_tokenizer, text_tokenizer, watermarker)
    
    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
        )
        audio = audio.squeeze().cpu().numpy()
    
    reset_caches(model)
    return audio


def validate(model, valloader, device, use_amp=True):
    """Validate the model on the validation set."""
    model.eval()
    val_losses = []
    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        for val_tokens, val_tokens_mask in valloader:
            val_tokens = val_tokens.to(device)
            val_tokens_mask = val_tokens_mask.to(device)
            val_loss = model(val_tokens, val_tokens_mask).item()
            val_losses.append(val_loss)
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss
