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
    timeout=14400,  # 4 hours should be plenty
    cpu=8.0,  # More CPUs for batch tokenization
    memory=32768,  # 32GB for larger batches
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def materialize_fineweb_edu(
    train_tokens: int = 20_000_000_000,
    val_tokens: int = 100_000_000,
    test_tokens: int = 100_000_000,
    sample_name: str = "sample-100BT",
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    shuffle_buffer_size: int = 10_000,
    shuffle_seed: int = 42,
    estimated_tokens_per_doc: int = 999,  # Pull this from get_avg_token_count_of_document
) -> None:
    """
    Download FineWeb-Edu and create train/val/test splits.

    Args:
        train_tokens: Training set size in tokens (default: 20B)
        val_tokens: Validation set size in tokens (default: 100M)
        test_tokens: Test set size in tokens (default: 100M)
        sample_name: FineWeb-Edu sample to use
        tokenizer_name: HuggingFace tokenizer to use
        shuffle_buffer_size: Shuffle buffer size
        shuffle_seed: Random seed
        estimated_tokens_per_doc: Average tokens per doc (from sampling)
    """
    # Setup cache
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = f"{VOL}/hf_home"
    os.environ["HF_HUB_CACHE"] = f"{os.environ['HF_HOME']}/hub"

    from datasets import load_dataset, Dataset, DatasetDict
    from transformers import AutoTokenizer

    total_target_tokens = train_tokens + val_tokens + test_tokens
    print(f"Target tokens: train={train_tokens:,} val={val_tokens:,} test={test_tokens:,} total={total_target_tokens:,}")

    # Estimate documents needed (with 20% buffer)
    estimated_docs = int((total_target_tokens / estimated_tokens_per_doc) * 1.2)
    print(f"Estimated documents needed: {estimated_docs:,} (based on {estimated_tokens_per_doc} tokens/doc)")

    # Load dataset
    print(f"Loading FineWeb-Edu {sample_name} in streaming mode...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=sample_name,
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR,
    )

    # Shuffle
    print(f"Shuffling with buffer size {shuffle_buffer_size:,}...")
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)

    # Download documents (fast, no tokenization yet)
    print(f"Downloading {estimated_docs:,} documents...")
    documents = []
    start_time = time.time()
    last_report = start_time

    for i, doc in enumerate(ds):
        documents.append(doc)

        current_time = time.time()
        if current_time - last_report >= 60:  # Report every minute
            elapsed = current_time - start_time
            docs_per_sec = (i + 1) / elapsed
            eta_seconds = (estimated_docs - i - 1) / docs_per_sec if docs_per_sec > 0 else 0
            print(f"Downloaded {i+1:,}/{estimated_docs:,} docs ({docs_per_sec:.1f} docs/sec, ETA: {eta_seconds/60:.1f}m)")
            last_report = current_time

        if i + 1 >= estimated_docs:
            break

    download_time = time.time() - start_time
    print(f"✓ Downloaded {len(documents):,} documents in {download_time:.1f}s ({len(documents)/download_time:.1f} docs/sec)")

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=CACHE_DIR,
        use_fast=True,
        trust_remote_code=True,
    )
    print(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Batch tokenize to count tokens and create splits
    print("Tokenizing documents and creating splits...")
    tokenize_start = time.time()

    train_docs = []
    val_docs = []
    test_docs = []
    train_tok = 0
    val_tok = 0
    test_tok = 0

    batch_size = 2000  # Larger batches for faster tokenization

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        texts = [d["text"] for d in batch]

        # Fast batch tokenization
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for j, doc in enumerate(batch):
            num_tokens = len(encoded["input_ids"][j])

            # Assign to splits in order: train -> val -> test
            if train_tok < train_tokens:
                train_docs.append(doc)
                train_tok += num_tokens
            elif val_tok < val_tokens:
                val_docs.append(doc)
                val_tok += num_tokens
            elif test_tok < test_tokens:
                test_docs.append(doc)
                test_tok += num_tokens
            else:
                # We have enough for all splits
                break

        if test_tok >= test_tokens:
            break

        if (i + batch_size) % 20000 == 0:
            print(f"Tokenized {i+batch_size:,}/{len(documents):,} docs | "
                  f"train: {train_tok:,}/{train_tokens:,} | "
                  f"val: {val_tok:,}/{val_tokens:,} | "
                  f"test: {test_tok:,}/{test_tokens:,}")

    tokenize_time = time.time() - tokenize_start
    total_time = time.time() - start_time

    print(f"\n✓ Tokenization complete in {tokenize_time:.1f}s")
    print(f"  Train: {len(train_docs):,} docs, {train_tok:,} tokens")
    print(f"  Val:   {len(val_docs):,} docs, {val_tok:,} tokens")
    print(f"  Test:  {len(test_docs):,} docs, {test_tok:,} tokens")
    print(f"  Total: {len(train_docs)+len(val_docs)+len(test_docs):,} docs, {train_tok+val_tok+test_tok:,} tokens")

    # Create datasets
    print("Creating dataset splits...")
    train_ds = Dataset.from_list(train_docs)
    val_ds = Dataset.from_list(val_docs)
    test_ds = Dataset.from_list(test_docs)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    # Save
    tokenizer_slug = tokenizer_name.replace("/", "_").replace("-", "_")
    save_path = f"{BASE_DIR}/{sample_name}_{train_tok//1_000_000}M_{tokenizer_slug}_with_splits"

    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path)

    # Create manifest
    meta = {
        "source": "HuggingFaceFW/fineweb-edu",
        "config": sample_name,
        "splits": {
            "train": {
                "target_tokens": train_tokens,
                "actual_tokens": train_tok,
                "num_documents": len(train_docs),
            },
            "validation": {
                "target_tokens": val_tokens,
                "actual_tokens": val_tok,
                "num_documents": len(val_docs),
            },
            "test": {
                "target_tokens": test_tokens,
                "actual_tokens": test_tok,
                "num_documents": len(test_docs),
            },
        },
        "total_tokens": train_tok + val_tok + test_tok,
        "total_documents": len(train_docs) + len(val_docs) + len(test_docs),
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": tokenizer.vocab_size,
            "estimated_tokens_per_doc": estimated_tokens_per_doc,
        },
        "shuffle_buffer_size": shuffle_buffer_size,
        "shuffle_seed": shuffle_seed,
        "paths": {
            "save_to_disk": save_path,
            "hf_cache": CACHE_DIR,
        },
        "timing": {
            "download_seconds": download_time,
            "tokenize_seconds": tokenize_time,
            "total_seconds": total_time,
        },
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }

    manifest_path = f"{BASE_DIR}/manifest_{sample_name}_{train_tok//1_000_000}M_{tokenizer_slug}_with_splits.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✓ Saved dataset with splits")
    print(f"✓ Path: {save_path}")
    print(f"✓ Manifest: {manifest_path}")
    print(f"✓ Total time: {total_time/60:.1f} minutes")

@app.function(
    image=image,
    volumes={VOL: vol},
    timeout=46800, # 13 hours, since we're downloading 20B tokens (4 hours is too short)
    cpu=4.0,
    secrets=[modal.Secret.from_name("huggingface-secret")]
    )

def get_avg_token_count_of_document(
    target_doc_count: int = 100_000,
    sample_name: str = "sample-100BT",
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    use_fast: bool = True,
    shuffle_buffer_size: int = 10_000,
    shuffle_seed: int = 42,
    ) -> None:
    """
    Check the average and median token count of a document in the dataset by processing a sample.

    Args:
        target_doc_count: The number of documents to process. (default: 100,000)
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
    token_counts = []
    total_tokens = 0
    doc_count = 0

    # Shuffle the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)

    print(f"Collecting documents until we hit {target_doc_count} documents...")
    start_time = time.time()
    last_report = start_time
    for doc in ds:
        text = doc["text"]

        # count tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)

        token_counts.append(num_tokens)
        total_tokens += num_tokens
        doc_count += 1

        # report progress every 5 minutes
        current_time = time.time()
        if current_time - last_report >= 300: # 5 minutes
            elapsed = current_time - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            eta_seconds = (target_doc_count - doc_count) / tokens_per_sec if tokens_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60

            print(f"Progress: {doc_count:,} docs | {total_tokens:,}/{target_doc_count:,} tokens "
                  f"({100* doc_count/target_doc_count:.1f}% | "
                  f"ETA: {eta_minutes:.1f} minutes)")
            last_report = current_time

        # stop if we hit target tokens
        if doc_count >= target_doc_count:
            break
    elapsed_total = time.time() - start_time
    print(f"Collected {doc_count:,} documents in {elapsed_total:.2f} seconds ({doc_count/elapsed_total:.2f} docs/sec)")

    #get mean, median of token counts
    avg_token_count = sum(token_counts) / len(token_counts)
    median_token_count = sorted(token_counts)[len(token_counts) // 2]
    print(f"Average token count: {avg_token_count:.2f}")
    print(f"Median token count: {median_token_count}")
    print(f"Total documents: {doc_count}")


@app.local_entrypoint()
def token_count_stats():
    get_avg_token_count_of_document.remote()

@app.local_entrypoint()
def main():
    materialize_fineweb_edu.remote()
    print(f"Volume '{VOLUME_NAME}' now contains fineweb-edu at {BASE_DIR}")