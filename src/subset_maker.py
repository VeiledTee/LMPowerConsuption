import json
import random

from datasets import Dataset, load_dataset

from src.config import CONFIG

n_total = 128

if "hotpot" in CONFIG.dataset_name:
    dataset: Dataset = load_dataset(
        CONFIG.dataset_name,
        CONFIG.config,
        split=CONFIG.split,
        trust_remote_code=True,
    )

    q_types = {}
    for q in dataset:
        t = q["type"]
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
else:  # bool q
    dataset: Dataset = load_dataset(
        CONFIG.dataset_name,
        split=CONFIG.split,
        trust_remote_code=True,
    )

    # Separate True and False samples
    true_samples = [ex for ex in dataset if ex["answer"] is True]
    false_samples = [ex for ex in dataset if ex["answer"] is False]

    # Calculate target counts (preserve 62.3% True / 37.7% False ratio)
    n_true = round(n_total * 0.623)  # ≈80
    n_false = n_total - n_true  # ≈48

    # Randomly sample each group
    subset_true = random.sample(true_samples, n_true)
    subset_false = random.sample(false_samples, n_false)

    # Combine and shuffle
    final_subset = subset_true + subset_false
    random.shuffle(final_subset)

    # Verification
    true_count = sum(1 for ex in final_subset if ex["answer"] is True)
    print(f"Final subset: {len(final_subset)} examples")
    print(f"True ratio: {true_count / n_total * 100:.1f}% ({true_count} samples)")
    print(
        f"False ratio: {(n_total - true_count) / n_total * 100:.1f}% ({n_total - true_count} samples)"
    )

# Save to hotpot_mini.jsonl
output_path = (
    CONFIG.data_dir / f"{CONFIG.dataset_name.split('/')[-1]}_mini_{n_total}.jsonl"
)
with open(output_path, "w", encoding="utf-8") as f:
    for example in final_subset:
        json.dump(example, f)
        f.write("\n")
