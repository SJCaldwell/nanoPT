import os
from pathlib import Path

import modal
import modal.experimental

# CUDA configuration
cuda_version = "12.6.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Paths
LOCAL_CODE_DIR = Path(__file__).parent.parent.absolute()
REMOTE_CODE_DIR = "/root/nanopt"
REMOTE_TRAIN_SCRIPT = "/root/nanopt/main.py"

GPU_TYPE = "H100"
DATASET_VOLUME_NAME = "hf-fineweb-edu"

# cluster configuration
single_node_gpus = 1
n_proc_per_node = 1

base_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .apt_install("git", "libibverbs-dev", "libibverbs1")
    # Install runtime deps from the project's pyproject.toml
    .pip_install_from_pyproject(str(LOCAL_CODE_DIR / "pyproject.toml"))
)

# Add local code to image
image = base_image.add_local_dir(
    LOCAL_CODE_DIR,
    remote_path="/root",
)

app = modal.App(name="nanopt-llama-3.2-1b-training", image=image)

# Modal volumes
training_volume = modal.Volume.from_name
