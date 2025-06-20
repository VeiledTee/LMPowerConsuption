from datasets import load_dataset, Dataset
import random
import json

from src.config import CONFIG

# Load and optionally sample dataset
dataset: Dataset = load_dataset(
    CONFIG.dataset_name,
    CONFIG.config,
    split=CONFIG.split,
    trust_remote_code=True,
)

# Shuffle and select first 128
dataset = dataset.shuffle(seed=42)
mini_dataset = dataset.select(range(128))

# Save to hotpot_mini.jsonl
output_path = CONFIG.result_dir / "hotpot_mini.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for example in mini_dataset:
        json.dump(example, f)
        f.write("\n")
