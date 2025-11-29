import argparse
import os
from dotenv import load_dotenv
import pickle
import yaml
from pathlib import Path
from tqdm import tqdm
import optuna
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import wandb

from utils import (
    load_model, 
    load_tokenizers, 
    generate_audio, 
    WarmupDecayLR,
    validate,
    load_watermarker,
    MIMI_SAMPLE_RATE,
)
from dataloaders import create_dataloaders

if os.getenv("WANDB_API_KEY") is None:
    raise ValueError("WANDB_API_KEY is not set in the .env file")


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/tokens.hdf5", type=str, help="Path to the pre-tokenized data")
    parser.add_argument("--output_dir", type=Path, default="./exp", help="Path to save the model")
    parser.add_argument("--config", type=str, default='./configs/finetune_param_defaults.yaml', help="Path to the finetuning config")
    parser.add_argument("--model_name_or_checkpoint_path", type=str, default="sesame/csm-1b", help="Pretrained model name or path to local checkpoint or huggingface model")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--partial_data_loading", action="store_true", help="Use partial data loading (use for large datasets)")

    parser.add_argument("--wandb_project", type=str, default="csm-finetuning", help="Name of the project")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name of the run")
    parser.add_argument("--wandb_reinit", type=bool, default=True, help="Whether to reinitialize the run")

    parser.add_argument("--log_every", type=int, default=10, help="Log every n steps")
    parser.add_argument("--val_every", type=int, default=100, help="Validate every n steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every n steps")
    parser.add_argument("--gen_every", type=int, default=1000, help="Generate every n steps")
    parser.add_argument(
        "--gen_sentences",
        type=str,
        default="Bird law in this country is not governed by reason.",
        help="Sentence(s) for model to generate. If a path is provided, the model will generate from the sentences in the file.",
    )
    parser.add_argument("--gen_speaker", type=int, default=999, help="Speaker id for model to generate")

    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs to train. If not provided, the training will run indefinitely.")

    args = parser.parse_args(arg_string.split() if arg_string else None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gen_sentences = Path(args.gen_sentences) if args.gen_sentences.endswith(".txt") else args.gen_sentences

    if args.train_from_scratch:
        args.model_name_or_checkpoint_path = None
    
    return args


def train(args: argparse.Namespace, config: dict, device: torch.device, trial: optuna.Trial = None):
    """
    trial is only used when we are sweeping hyperparameters.
    """
    assert wandb.run is not None, "Wandb is not initialized"

    eff_batch_size = config["batch_size"] * config["grad_acc_steps"]
    
    # Load / create: model, tokenizers, dataloaders, optimizer, scheduler, and grad scaler.
    model = load_model(model_name_or_checkpoint_path=args.model_name_or_checkpoint_path, device=device, decoder_loss_weight=config["decoder_loss_weight"])
    text_tokenizer, audio_tokenizer = load_tokenizers(device)
    watermarker = load_watermarker(device=device)
    trainloader, valloader = create_dataloaders(
        args.data, 
        config["batch_size"], 
        infinite_train=False,
        load_in_memory=not args.partial_data_loading
    )
    total_steps = args.n_epochs * len(trainloader) if args.n_epochs else None
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = WarmupDecayLR(optimizer, config["warmup_steps"], total_steps, config["lr_decay"])
    scaler = GradScaler(enabled=args.use_amp)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "effective_batch_size": eff_batch_size,
        "config": config,
        "args": args,
        "best_val_loss": float("inf"),
    }
    
    # Training loop
    step = 0
    train_losses = []
    pbar = tqdm(total=total_steps, desc="Training" if trial is None else f"Trial {trial.number}")
    model.train()
    
    for epoch in range(args.n_epochs):
        for tokens, tokens_mask in trainloader:
            tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)
                
            with autocast(device_type=str(device), enabled=args.use_amp):
                loss = model(tokens, tokens_mask)
                loss = loss / config["grad_acc_steps"]
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config["grad_acc_steps"] == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss = loss.item()
            train_losses.append(train_loss)
            
            if args.log_every and step % args.log_every == 0:
                wandb.log(
                    {
                        "train_loss_avg": sum(train_losses) / len(train_losses),
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
                train_losses = []

            if args.save_every and (step % args.save_every == 0 or step == total_steps - 1):
                state["model"] = model.state_dict()
                torch.save(state, args.output_dir / f"model_{step}.pt")
                if step == total_steps - 1:
                    torch.save(state, args.output_dir / f"model_final.pt")

            if args.val_every and (step % args.val_every == 0 or step == total_steps - 1):
                val_loss = validate(model, valloader, device, args.use_amp)
                wandb.log({"val_loss": val_loss}, step=step)

                if val_loss < state["best_val_loss"]:
                    state["best_val_loss"] = val_loss
                    torch.save(state, args.output_dir / f"model_bestval.pt")
                    wandb.save(args.output_dir / f"wandb_bestval.pt")

                # If this finetune is part of a sweep, report the validation loss to Optuna for pruning
                if trial is not None:
                    trial.report(val_loss, step)
                    if trial.should_prune():
                        wandb.finish()
                        pbar.close()
                        raise optuna.exceptions.TrialPruned()
                
                model.train()
                pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
            else:
                pbar.set_postfix(
                    {"train_loss": f"{train_loss:.4f}", "learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch}
                )
            
            if args.gen_every and step % args.gen_every == 0 and not (args.train_from_scratch and step == 0):

                gen_sentences = []
                if isinstance(args.gen_sentences, str):
                    gen_sentences.append(args.gen_sentences)
                elif isinstance(args.gen_sentences, Path):
                    with open(args.gen_sentences, "r") as f:
                        gen_sentences = f.readlines()

                for i, sentence in enumerate(gen_sentences):
                    audio = generate_audio(
                        model,
                        audio_tokenizer,
                        text_tokenizer,
                        watermarker,
                        sentence,
                        args.gen_speaker,
                        device,
                        use_amp=args.use_amp
                    )
                    
                    wandb.log({f"audio_{i}": wandb.Audio(audio, sample_rate=MIMI_SAMPLE_RATE)}, step=step)
                model.train()
            
            pbar.update(1)
            if step >= total_steps:
                break
            
            step += 1
    
    pbar.close()
    return state["best_val_loss"]


if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"training_bs-{config['batch_size']}x{config['grad_acc_steps']}",
        notes=f"Config: {config}",
        config={**config, **vars(args)},
        reinit=args.wandb_reinit,
        dir=args.output_dir / "wandb",
    )

    final_val_loss = train(args, config, device)

    wandb.finish()
