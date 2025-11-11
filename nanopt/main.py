import wandb
from cyclopts import App
from nanopt.common import ddp_setup, set_seed_all
from nanopt.models.llama import LlamaForCausalLM, LlamaConfig
from nanopt.data import get_dataloaders
from nanopt.optimizers.adamw import create_optimizer, get_lr_scheduler
import torch.distributed as dist
import os
import torch

app = App("nanopt")

@app.default
def train_model(
    seed: int = 42,
    project_name: str = "nanopt-llama-3.2-1b-pretrain",
    learning_rate: float = 3e-4,
    warmup_steps: int = 1000,
    max_steps: int = 10000,
    per_device_batch_size: int = 16,
    batch_size: int = 16,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    max_length: int = 4096,
    num_workers: int = 4,
    dataset_path: str = "data/fineweb-edu",
) -> None:
    set_seed_all(seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    assert batch_size % per_device_batch_size == 0 # check even gradient accumulation
    gradient_accumulation_steps = batch_size / per_device_batch_size

    if global_rank == 0:
        #put all the config in here, which should be everything passed to the app. 
        config = {
        }
        wandb.init(project=project_name)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = LlamaForCausalLM(config)
    model.to(device)
    optimizer = create_optimizer(model, lr=learning_rate)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)

    train_dataloader, val_dataloader = get_dataloaders(
        dataset_path=dataset_path,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
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
                if i % checkpoint_interval == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=real_step,
                        loss=loss.item(),
                    )
    if global_rank == 0:
        wandb.finish()

def main():
    print("Training NanoPT...")
    ddp_setup()
    app()
    dist.destroy_process_group()