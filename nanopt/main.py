import wandb
from cyclopts import App
from nanopt.common import ddp_setup, set_seed_all, save_checkpoint
from nanopt.models.llama import LlamaForCausalLM, LlamaConfig
from nanopt.data import get_dataloaders
from nanopt.optimizers.adamw import create_optimizer, get_lr_scheduler
import torch.distributed as dist
import os
import torch
from dataclasses import dataclass

app = App("nanopt")

@dataclass
class TrainConfig:
    seed: int = 42
    project_name: str = "nanopt-llama-3.2-1b-pretrain"
    checkpoint_dir: str = "checkpoints"
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 10000
    per_device_batch_size: int = 16 # tune based on GPU size
    tokens_per_batch: int = 1_000_000 # million seems to be the pretraining token batch size
    tokenizer_name: str = "meta-llama/Llama-3.2-1B"
    sequence_length: int = 4096
    num_workers: int = 4
    dataset_path: str = "data/fineweb-edu"
    checkpoint_interval: int = 1000

@app.default
def train_model(
    config: TrainConfig,
) -> None:
    set_seed_all(config.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    if world_size > 1 and not torch.cuda.is_available():
        raise ValueError("Distributed training requires CUDA")

    # get total number of gpus in the world (assume homogeneity)

    gradient_accumulation_steps = config.tokens_per_batch // (config.per_device_batch_size * world_size * config.max * config.sequence_length)

    if global_rank == 0:
        #put all the config in here, which should be everything passed to the app. 
        wandb_config = {
            **config.__dict__,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
        wandb.init(project=config.project_name, config=wandb_config)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    ))
    model.to(device)
    optimizer = create_optimizer(model, lr=config.learning_rate)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=config.warmup_steps, max_steps=config.max_steps)

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
        output.loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if global_rank == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                })
                if i % config.checkpoint_interval == 0:
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
    print("Training NanoPT...")
    #ddp_setup()
    app()
    #dist.destroy_process_group()

if __name__ == "__main__":
    main()