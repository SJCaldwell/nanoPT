import os
import json
import time
import modal

app = modal.App(name="hf-fineweb-edu-volume")

VOLUME_NAME = "hf-fineweb-edu"

vol = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "datasets",
        "huggingface-hub",
        "pyarrow",
        "tqdm",
        "xxhash",
        "zstandard",
        "transformers"
    )
)

VOL = "/vol"
BASE_DIR = f"{VOL}/datasets/fineweb-edu"
CACHE_DIR = f"{VOL}/hf_cache"

@app.function(
    image=image,
    volumes={VOL: vol},
    timeout=14400, # 4 hours, since we're downloading 20B tokens
    cpu=4.0,
    secrets=[modal.Secret.from_name("huggingface-secret")]
    )
def materialize_fineweb_edu(
    target_tokens: int = 20_000_000_000,
    sample_name: str = "sample-100BT",
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    use_fast: bool = True,
    shuffle_buffer_size: int = 10_000,
    shuffle_seed: int = 42,
    ) -> None:
    """
    Download FineWeb-Edu until we hit target_tokens.

    Args:
        target_tokens: The number of tokens to download. (default: 20B)
        sample_name: The name of the sample to download. (default: "sample-100BT")
        tokenizer_name: HuggingFace model/tokenizer identifier or local path
        use_fast: Whether to use the fast tokenizer. (default: True)
        shuffle_buffer_size: Size of the shuffle buffer. (default: 10,000)
        shuffle_seed: Random seed for shuffling. (default: 42)
    """
    #setup cache
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = f"{VOL}/hf_home"
    os.environ["HF_HUB_CACHE"] = f"{os.environ['HF_HOME']}/hub"

    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=CACHE_DIR,
            use_fast=use_fast,
            trust_remote_code=True,
        )
        print(f"Loaded tokenizer with a vocab size of {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise e

    print(f"Loading FineWeb-Edu {sample_name} in streaming mode...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=sample_name,
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR,
    )
    # collect documents until we hit target token count
    documents = []
    total_tokens = 0
    doc_count = 0

    # Shuffle the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
    print(f"Shuffled dataset with {len(ds)} documents")

    print(f"Collecting documents until we hit {target_tokens} tokens...")
    start_time = time.time()
    last_report = start_time
    for doc in ds:
        text = doc["text"]

        # count tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)

        documents.append(doc)
        total_tokens += num_tokens
        doc_count += 1

        # report progress every 5 minutes
        current_time = time.time()
        if current_time - last_report >= 300: # 5 minutes
            elapsed = current_time - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            eta_seconds = (target_tokens - total_tokens) / tokens_per_sec if tokens_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60

            print(f"Progress: {doc_count:,} docs | {total_tokens:,}/{target_tokens:,} tokens "
                  f"({100* total_tokens/target_tokens:.1f}% | "
                  f"ETA: {eta_minutes:.1f} minutes)")
            last_report = current_time

        # stop if we hit target tokens
        if total_tokens >= target_tokens:
            break
    elapsed_total = time.time() - start_time
    print(f"Collected {doc_count:,} documents in {elapsed_total:.2f} seconds ({total_tokens/elapsed_total:.2f} tokens/sec)")

    print("Converting to dataset...")

    final_ds = Dataset.from_list(documents)
    # convert documents to a single dataset
    print(f"Converting {doc_count} documents to a single dataset...")
    ds = Dataset.from_list(documents)
    print(f"Saved {len(ds)} documents to {BASE_DIR}")

    tokenizer_slug = tokenizer_name.replace("/", "_").replace("-", "_")
    save_path = f"{BASE_DIR}/{sample_name}_{total_tokens//1_000_000}M_{tokenizer_slug}_shuffled"

    print(f"Saving to {save_path}...")
    final_ds.save_to_disk(save_path)

    # Create manifest
    meta = {
        "source": "HuggingFaceFW/fineweb-edu",
        "config": sample_name,
        "target_tokens": target_tokens,
        "actual_tokens": total_tokens,
        "num_documents": doc_count,
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": tokenizer.vocab_size,
            "use_fast": use_fast,
        },
        "shuffle_buffer_size": shuffle_buffer_size,
        "shuffle_seed": shuffle_seed,
        "paths": {
            "save_to_disk": save_path,
            "hf_cache": CACHE_DIR
        },
        "processing_time_seconds": elapsed_total,
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }

    manifest_path = f"{BASE_DIR}/manifest_{sample_name}_{total_tokens//1_000_000}M_{tokenizer_slug}_shuffled.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=2)

@app.local_entrypoint()
def main():
    materialize_fineweb_edu.remote()
    print(f"Volume '{VOLUME_NAME}' now contains fineweb-edu at {BASE_DIR}")