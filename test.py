from datasets import load_dataset, Dataset
import random
import json
from collections import defaultdict

from src.config import CONFIG

dataset: Dataset = load_dataset(
    CONFIG.dataset_name,
    CONFIG.config,
    split=CONFIG.split,
    trust_remote_code=True,
)

n_total = 128

q_types = {}
for q in dataset:
    t = q['type']
    q_types[t] = q_types.get(t, 0) + 1

print(q_types)

bridge_qs = [q for q in dataset if q["type"] == "bridge"]
comp_qs = [q for q in dataset if q["type"] == "comparison"]

print(len(bridge_qs))
print(len(comp_qs))

subset_bridge = random.sample(bridge_qs, 102)  # 80%
subset_comp = random.sample(comp_qs, 26)  # 20%

final_subset = subset_bridge + subset_comp
random.shuffle(final_subset)

types = [q["type"] for q in final_subset]
print(types.count("bridge") / n_total * 100)
print(types.count("comparison") / n_total * 100)

# Save to hotpot_mini.jsonl
output_path = CONFIG.result_dir / f"hotpot_mini_{n_total}.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for example in final_subset:
        json.dump(example, f)
        f.write("\n")
