import wandb
import yaml
from nanopt.common import ddp_setup, set_seed_all, save_checkpoint
from nanopt.models.llama import LlamaForCausalLM, LlamaConfig
from nanopt.data import get_dataloaders
from nanopt.optimizers.adamw import create_optimizer, get_lr_scheduler
import torch.distributed as dist
import os
import torch
from pydantic import BaseModel
import sys
import math


class TrainConfig(BaseModel):
    seed: int = 42
    project_name: str = "nanopt-llama-3.2-1b-pretrain"
    checkpoint_dir: str = "checkpoints"
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    per_device_batch_size: int = 16 # tune based on GPU size
    tokens_per_batch: int = 1_000_000 # million seems to be the pretraining token batch size
    tokenizer_name: str = "meta-llama/Llama-3.2-1B"
    sequence_length: int = 4096
    num_workers: int = 4
    dataset_path: str = "data/fineweb-edu"
    checkpoint_interval: int = 1000

def train_model(
    config: TrainConfig,
) -> None:
    set_seed_all(config.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    print(config)

    if world_size > 1 and not torch.cuda.is_available():
        raise ValueError("Distributed training requires CUDA")

    # get total number of gpus in the world (assume homogeneity)

    gradient_accumulation_steps = config.tokens_per_batch // (config.per_device_batch_size * world_size * config.sequence_length)
    # need to calculate max steps based on target number of tokens to give to the lr scheduler
    max_steps = 20_000_000_000 // config.tokens_per_batch

    if global_rank == 0:
        #put all the config in here, which should be everything passed to the app. 
        wandb_config = {
            **config.__dict__,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
        wandb.init(project=config.project_name, config=wandb_config)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = LlamaForCausalLM(LlamaConfig()) # default config is for Llama 3.2 1B
    model.to(device)
    optimizer = create_optimizer(model, lr=config.learning_rate)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=config.warmup_steps, max_steps=max_steps)

    train_dataloader, val_dataloader = get_dataloaders(
        dataset_path=config.dataset_path,
        tokenizer_name=config.tokenizer_name,
        batch_size=config.per_device_batch_size,
        max_length=config.sequence_length,
        num_workers=config.num_workers,
        world_size=world_size,
        rank=global_rank,
    )
    model.train()

    for (i, batch) in enumerate(train_dataloader):
        real_step = (i + 1) // gradient_accumulation_steps
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = output["loss"] / gradient_accumulation_steps
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if global_rank == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "global_step": i,
                    "real_step": real_step,
                })
                if real_step % config.checkpoint_interval == 0:
                    save_checkpoint(
                        checkpoint_dir=config.checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler, # type: ignore
                        step=real_step,
                        loss=loss.item(),
                    )
    
    if global_rank == 0:
        wandb.finish()

def main():
    print("Validating config...")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            train_config = TrainConfig.model_validate(config)
    else:
        train_config = TrainConfig()
    ddp_setup()
    train_model(train_config)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()