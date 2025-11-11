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
    timeout=28800,  # 8 hours should be plenty
    cpu=8.0,  # More CPUs for batch tokenization
    memory=32768,  # 32GB for larger batches
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def tokenize_existing_dataset(
    dataset_name: str = "fineweb-edu",
    dataset_local_path: str = f"{BASE_DIR}/fineweb-edu",
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    use_fast: bool = True,
    shuffle_buffer_size: int = 10_000,
    shuffle_seed: int = 42,
    estimated_tokens_per_doc: int = 999,  # Pull this from get_avg_token_count_of_document
) -> None:
    """
    Tokenize an existing dataset.

    Args:
        dataset_name: The name of the dataset to tokenize. (default: "fineweb-edu")
        dataset_local_path: The local path to the dataset to tokenize. (default: f"{BASE_DIR}/fineweb-edu")
        tokenizer_name: HuggingFace tokenizer to use. (default: "meta-llama/Llama-3.2-1B")
    """
    raise NotImplementedError("Not implemented yet")
    # some code to finish later
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
    timeout=28800,  # 8 hours
    cpu=8.0,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    retries=0,
)
def materialize_fineweb_edu_untokenized(
    train_tokens: int = 20_000_000_000,
    val_tokens: int = 100_000_000,
    test_tokens: int = 100_000_000,
    sample_name: str = "sample-100BT",
    shuffle_buffer_size: int = 10_000,
    shuffle_seed: int = 42,
    estimated_tokens_per_doc: int = 999,
    chunk_size: int = 100_000,
    checkpoint_interval: int = 300,  # Save checkpoint every 5 minutes
) -> None:
    """
    Download FineWeb-Edu and create train/val/test splits without tokenization.
    Memory-efficient version with checkpointing for resumability.
    """
    # Setup cache
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = f"{VOL}/hf_home"
    os.environ["HF_HUB_CACHE"] = f"{os.environ['HF_HOME']}/hub"

    from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
    import tempfile
    import shutil

    total_target_tokens = train_tokens + val_tokens + test_tokens
    print(f"Target tokens: train={train_tokens:,} val={val_tokens:,} test={test_tokens:,} total={total_target_tokens:,}")

    # Estimate documents needed (with 25% buffer)
    estimated_docs = int((total_target_tokens / estimated_tokens_per_doc) * 1.25)
    print(f"Estimated documents needed: {estimated_docs:,} (based on {estimated_tokens_per_doc} tokens/doc)")

    # Calculate split fractions
    train_frac = train_tokens / total_target_tokens
    val_frac = val_tokens / total_target_tokens
    test_frac = test_tokens / total_target_tokens
    
    print(f"Split fractions: train={train_frac:.4f} ({train_frac*100:.2f}%), "
          f"val={val_frac:.4f} ({val_frac*100:.2f}%), "
          f"test={test_frac:.4f} ({test_frac*100:.2f}%)")
    
    # Calculate document counts for each split
    train_count = int(estimated_docs * train_frac)
    val_count = int(estimated_docs * val_frac)
    test_count = int(estimated_docs * test_frac)
    
    print(f"Target document counts: train={train_count:,}, val={val_count:,}, test={test_count:,}")

    # Setup checkpoint and temp directories
    checkpoint_dir = f"{BASE_DIR}/.checkpoints/{sample_name}_{shuffle_seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/progress.json"
    
    train_temp = f"{checkpoint_dir}/train"
    val_temp = f"{checkpoint_dir}/val"
    test_temp = f"{checkpoint_dir}/test"
    os.makedirs(train_temp, exist_ok=True)
    os.makedirs(val_temp, exist_ok=True)
    os.makedirs(test_temp, exist_ok=True)

    # Check for existing checkpoint
    resume_from = None
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Found existing checkpoint at {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Verify checkpoint matches current run parameters
        if (checkpoint_data.get("sample_name") == sample_name and
            checkpoint_data.get("shuffle_seed") == shuffle_seed and
            checkpoint_data.get("estimated_docs") == estimated_docs):
            
            resume_from = checkpoint_data
            print(f"✓ Resuming from document {resume_from['doc_count']:,}")
            print(f"  Train chunks: {resume_from['train_saved']}, Val chunks: {resume_from['val_saved']}, Test chunks: {resume_from['test_saved']}")
        else:
            print("⚠ Checkpoint parameters don't match, starting fresh")
            # Clean up old checkpoint
            shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(train_temp, exist_ok=True)
            os.makedirs(val_temp, exist_ok=True)
            os.makedirs(test_temp, exist_ok=True)

    # Load dataset
    print(f"\nLoading FineWeb-Edu {sample_name} in streaming mode...")
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

    # Initialize counters
    if resume_from:
        train_saved = resume_from['train_saved']
        val_saved = resume_from['val_saved']
        test_saved = resume_from['test_saved']
        doc_count = resume_from['doc_count']
        start_doc = doc_count
        print(f"Skipping first {start_doc:,} documents...")
    else:
        train_saved = 0
        val_saved = 0
        test_saved = 0
        doc_count = 0
        start_doc = 0

    train_docs = []
    val_docs = []
    test_docs = []
    
    start_time = time.time()
    last_report = start_time
    last_checkpoint = start_time
    
    def save_chunk(docs, split_dir, split_name, saved_count):
        """Save a chunk of documents to disk."""
        if len(docs) == 0:
            return saved_count
        
        chunk_ds = Dataset.from_list(docs)
        chunk_path = f"{split_dir}/chunk_{saved_count:06d}"
        chunk_ds.save_to_disk(chunk_path)
        print(f"  Saved {split_name} chunk {saved_count} ({len(docs):,} docs)")
        return saved_count + 1
    
    def save_checkpoint():
        """Save progress checkpoint."""
        checkpoint_data = {
            "sample_name": sample_name,
            "shuffle_seed": shuffle_seed,
            "estimated_docs": estimated_docs,
            "doc_count": doc_count,
            "train_saved": train_saved,
            "val_saved": val_saved,
            "test_saved": test_saved,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        vol.commit()  # Commit volume to persist checkpoint
    
    # Download and split documents in chunks
    print(f"\nDownloading and splitting documents (chunk_size={chunk_size:,})...")
    
    for i, doc in enumerate(ds):
        # Skip documents we've already processed
        if i < start_doc:
            continue
        
        doc_count = i + 1
        
        # Assign to appropriate split
        if len(train_docs) + train_saved * chunk_size < train_count:
            train_docs.append(doc)
            if len(train_docs) >= chunk_size:
                train_saved = save_chunk(train_docs, train_temp, "train", train_saved)
                train_docs = []
        elif len(val_docs) + val_saved * chunk_size < val_count:
            val_docs.append(doc)
            if len(val_docs) >= chunk_size:
                val_saved = save_chunk(val_docs, val_temp, "val", val_saved)
                val_docs = []
        elif len(test_docs) + test_saved * chunk_size < test_count:
            test_docs.append(doc)
            if len(test_docs) >= chunk_size:
                test_saved = save_chunk(test_docs, test_temp, "test", test_saved)
                test_docs = []
        else:
            # We have enough for all splits
            print(f"\n✓ Collected all required documents!")
            break
        
        current_time = time.time()
        
        # Save checkpoint at regular intervals
        if current_time - last_checkpoint >= checkpoint_interval:
            print(f"💾 Saving checkpoint at document {doc_count:,}...")
            save_checkpoint()
            last_checkpoint = current_time
        
        # Progress reporting
        if current_time - last_report >= 60:
            elapsed = current_time - start_time
            docs_per_sec = (doc_count - start_doc) / elapsed if elapsed > 0 else 0
            eta_seconds = (estimated_docs - doc_count) / docs_per_sec if docs_per_sec > 0 else 0
            print(f"Downloaded {doc_count:,}/{estimated_docs:,} docs ({docs_per_sec:.1f} docs/sec, ETA: {eta_seconds/60:.1f}m)")
            print(f"  Current: train={len(train_docs) + train_saved*chunk_size:,}, "
                  f"val={len(val_docs) + val_saved*chunk_size:,}, "
                  f"test={len(test_docs) + test_saved*chunk_size:,}")
            last_report = current_time
    
    # Save remaining documents
    print("\nSaving final chunks...")
    if train_docs:
        train_saved = save_chunk(train_docs, train_temp, "train", train_saved)
    if val_docs:
        val_saved = save_chunk(val_docs, val_temp, "val", val_saved)
    if test_docs:
        test_saved = save_chunk(test_docs, test_temp, "test", test_saved)
    
    # Save final checkpoint
    save_checkpoint()
    
    download_time = time.time() - start_time
    print(f"\n✓ Downloaded {doc_count:,} documents in {download_time:.1f}s ({doc_count/download_time:.1f} docs/sec)")

    # Load all chunks and concatenate
    print("\nLoading and concatenating chunks...")
    concat_start = time.time()
    
    def load_split_from_chunks(split_dir):
        """Load all chunks from a split directory."""
        chunk_paths = sorted([f for f in os.listdir(split_dir) if os.path.isdir(f"{split_dir}/{f}")])
        if not chunk_paths:
            return Dataset.from_list([])
        
        print(f"  Loading {len(chunk_paths)} chunks...")
        datasets = []
        for chunk_path in chunk_paths:
            chunk_ds = Dataset.load_from_disk(f"{split_dir}/{chunk_path}")
            datasets.append(chunk_ds)
        
        return concatenate_datasets(datasets)
    
    print("Loading train split...")
    train_ds = load_split_from_chunks(train_temp)
    print(f"  Train: {len(train_ds):,} documents")
    
    print("Loading val split...")
    val_ds = load_split_from_chunks(val_temp)
    print(f"  Val: {len(val_ds):,} documents")
    
    print("Loading test split...")
    test_ds = load_split_from_chunks(test_temp)
    print(f"  Test: {len(test_ds):,} documents")
    
    concat_time = time.time() - concat_start
    
    # Create final dataset dict
    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })
    
    # Save final dataset
    save_path = f"{BASE_DIR}/{sample_name}_raw_{doc_count//1_000_000}M_docs"
    print(f"\nSaving final dataset to {save_path}...")
    save_start = time.time()
    dataset_dict.save_to_disk(save_path)
    save_time = time.time() - save_start
    
    # Clean up checkpoint directory
    print("Cleaning up checkpoint directory...")
    shutil.rmtree(checkpoint_dir)
    
    total_time = time.time() - start_time

    # Create manifest
    meta = {
        "source": "HuggingFaceFW/fineweb-edu",
        "config": sample_name,
        "format": "raw_text",
        "target_tokens": {
            "train": train_tokens,
            "validation": val_tokens,
            "test": test_tokens,
            "total": total_target_tokens,
        },
        "estimated_tokens_per_doc": estimated_tokens_per_doc,
        "splits": {
            "train": {
                "num_documents": len(train_ds),
                "estimated_tokens": len(train_ds) * estimated_tokens_per_doc,
            },
            "validation": {
                "num_documents": len(val_ds),
                "estimated_tokens": len(val_ds) * estimated_tokens_per_doc,
            },
            "test": {
                "num_documents": len(test_ds),
                "estimated_tokens": len(test_ds) * estimated_tokens_per_doc,
            },
        },
        "total_documents": doc_count,
        "shuffle_buffer_size": shuffle_buffer_size,
        "shuffle_seed": shuffle_seed,
        "chunk_size": chunk_size,
        "paths": {
            "save_to_disk": save_path,
            "hf_cache": CACHE_DIR,
        },
        "timing": {
            "download_seconds": download_time,
            "concatenate_seconds": concat_time,
            "save_seconds": save_time,
            "total_seconds": total_time,
        },
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }

    manifest_path = f"{BASE_DIR}/manifest_{sample_name}_raw_{doc_count//1_000_000}M_docs.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✓ Saved raw dataset with splits")
    print(f"✓ Path: {save_path}")
    print(f"✓ Manifest: {manifest_path}")
    print(f"✓ Download time: {download_time/60:.1f} minutes")
    print(f"✓ Concatenate time: {concat_time/60:.1f} minutes")
    print(f"✓ Save time: {save_time/60:.1f} minutes")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    
    vol.commit()  # Final commit


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
    materialize_fineweb_edu_untokenized.remote()
    print(f"Volume '{VOLUME_NAME}' now contains fineweb-edu at {BASE_DIR}")