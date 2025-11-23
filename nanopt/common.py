import random
import numpy as np
import torch
import os
from pathlib import Path
from typing import Dict, Optional
import torch.distributed as dist

def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "torch.float32":
        return torch.float32
    elif dtype_str == "torch.float16":
        return torch.float16
    elif dtype_str == "torch.bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype: {dtype_str}")

def set_seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # save as 'latest' checkpoint (overwrite)
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint (loss={loss:.4f}) to {best_path}")

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Loaded checkpoint at step {step} with loss {loss:.4f}")
    return step

def get_gpu_memory() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    gpu_memory = {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
    }
    return gpu_memory

def ddp_setup() -> None:
    # check if nccl is available
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
