import json
import random

from datasets import Dataset, load_dataset

from src.config import CONFIG

N_SAMPLES = CONFIG.n_samples or 128


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            json.dump(example, f)
            f.write("\n")


def process_hotpot(dataset: Dataset) -> list:
    q_types = {}
    for q in dataset:
        q_types[q["type"]] = q_types.get(q["type"], 0) + 1
    print(q_types)

    bridge_qs = [q for q in dataset if q["type"] == "bridge"]
    comp_qs = [q for q in dataset if q["type"] == "comparison"]

    print(f"bridge: {len(bridge_qs)}, comparison: {len(comp_qs)}")

    n_bridge = round(N_SAMPLES * 0.8)
    n_comp = N_SAMPLES - n_bridge

    subset = random.sample(bridge_qs, n_bridge) + random.sample(comp_qs, n_comp)
    random.shuffle(subset)

    print(
        f"bridge %: {sum(q['type'] == 'bridge' for q in subset) / N_SAMPLES * 100:.1f}%"
    )
    print(
        f"comparison %: {sum(q['type'] == 'comparison' for q in subset) / N_SAMPLES * 100:.1f}%"
    )

    return subset


def process_boolq(dataset: Dataset) -> list:
    true_qs = [ex for ex in dataset if ex["answer"] is True]
    false_qs = [ex for ex in dataset if ex["answer"] is False]

    n_true = round(N_SAMPLES * 0.623)
    n_false = N_SAMPLES - n_true

    subset = random.sample(true_qs, n_true) + random.sample(false_qs, n_false)
    random.shuffle(subset)

    true_count = sum(ex["answer"] is True for ex in subset)
    print(f"Final subset: {len(subset)} examples")
    print(f"True ratio: {true_count / N_SAMPLES * 100:.1f}% ({true_count})")
    print(
        f"False ratio: {(N_SAMPLES - true_count) / N_SAMPLES * 100:.1f}% ({N_SAMPLES - true_count})"
    )

    return subset


def main():
    if "hotpot" in CONFIG.dataset_name:
        dataset = load_dataset(
            CONFIG.dataset_name,
            CONFIG.config,
            split=CONFIG.split,
            trust_remote_code=True,
        )
        subset = process_hotpot(dataset)
    else:
        dataset = load_dataset(
            CONFIG.dataset_name, split=CONFIG.split, trust_remote_code=True
        )
        subset = process_boolq(dataset)

    output_path = (
        CONFIG.data_dir
        / f"{CONFIG.dataset_name.split('/')[-1]}_mini_dev_{N_SAMPLES}.jsonl"
    )
    save_jsonl(subset, output_path)


if __name__ == "__main__":
    main()
