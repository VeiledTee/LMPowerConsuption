from datasets import load_dataset, Dataset
import random
import json
from collections import defaultdict

from src.config import CONFIG

# Load dataset
dataset: Dataset = load_dataset(
    CONFIG.dataset_name,
    CONFIG.config,
    split=CONFIG.split,
    trust_remote_code=True,
)

# Group examples by question type
type_to_examples = defaultdict(list)
for example in dataset:
    q_type = example.get("type", "unknown")
    type_to_examples[q_type].append(example)

# Set sample sizes (approx 60/40 split)
random.seed(42)
n_total = 128
n_bridge = int(n_total * 0.6)
n_comparison = n_total - n_bridge
print(f"Bridge questions: {n_bridge}\nComparison questions: {n_comparison}\n=====\nTotal questions:  {n_bridge + n_comparison}")

# Sample from each type
sampled_bridge = random.sample(type_to_examples["bridge"], n_bridge)
sampled_comparison = random.sample(type_to_examples["comparison"], n_comparison)

# Combine and shuffle
sampled_data = sampled_bridge + sampled_comparison
random.shuffle(sampled_data)

# Save to hotpot_mini.jsonl
output_path = CONFIG.result_dir / "hotpot_mini.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for example in sampled_data:
        json.dump(example, f)
        f.write("\n")
