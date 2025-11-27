import json
import random
import math
import pandas as pd

from datasets import Dataset, load_dataset

from src.config import CONFIG

# Use N_SAMPLES from CONFIG, defaulting to 128
N_SAMPLES = CONFIG.dataset_size or 128


def save_jsonl(data: list, path: str):
    """Saves a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            json.dump(example, f)
            f.write("\n")


# ----------------------------------------------------------------------
# Helper Function for Dynamic Stratified Counts
# ----------------------------------------------------------------------
def calculate_stratified_counts(
    full_counts: dict, subset_size: int, total_full_samples: int
) -> dict:
    """Calculates the proportional counts for the subset, ensuring the total is exact."""

    float_targets = {}

    # 1. Calculate the float target size for each type
    for q_type, count in full_counts.items():
        ratio = count / total_full_samples
        float_targets[q_type] = ratio * subset_size

    # 2. Round down all counts (initial floor)
    target_counts_floored = {k: math.floor(v) for k, v in float_targets.items()}

    # 3. Determine how many remaining samples are needed
    remaining_needed = subset_size - sum(target_counts_floored.values())

    # 4. Use fractional parts for tie-breaking to allocate remaining samples
    fractional_parts = {
        k: v - target_counts_floored[k] for k, v in float_targets.items()
    }
    sorted_fractions = sorted(
        fractional_parts.items(), key=lambda item: item[1], reverse=True
    )

    final_counts = target_counts_floored

    # Assign the remaining needed samples one by one
    for i in range(remaining_needed):
        q_type_to_add = sorted_fractions[i][0]
        final_counts[q_type_to_add] += 1

    return final_counts


# ----------------------------------------------------------------------
# Processor Functions
# ----------------------------------------------------------------------
def process_2wikimultihop_local(file_path: str) -> list:
    """
    Loads data from a local JSON file and creates a stratified subset
    based on the 'type' distribution.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df_full = pd.DataFrame(data)

    except FileNotFoundError:
        print(f"ERROR: Local file not found at '{file_path}'.")
        print("Please check CONFIG.wiki_file_path.")
        return []

    # Dynamically get the full counts and total size from the loaded file
    full_counts = df_full["type"].value_counts().to_dict()
    total_full_samples = len(df_full)

    print("Full Dataset Question Type Distribution:")
    print(pd.Series(full_counts).to_string())

    # Calculate the required proportional counts for the subset size
    target_counts = calculate_stratified_counts(
        full_counts, N_SAMPLES, total_full_samples
    )

    print("-" * 50)
    print(f"Target Stratified Counts for N_SAMPLES={N_SAMPLES}:")
    print(pd.Series(target_counts).to_string())
    print("-" * 50)

    # Perform stratified sampling
    sampled_data = []

    for q_type, count in target_counts.items():
        df_type = df_full[df_full["type"] == q_type]

        if len(df_type) < count:
            print(
                f"Warning: Not enough samples for '{q_type}'. Only sampling available {len(df_type)}."
            )
            sample = df_type
        else:
            # Use random_state=42 for reproducibility
            sample = df_type.sample(n=count, random_state=42)

        sampled_data.append(sample)

    # Combine and shuffle the subset
    df_subset = pd.concat(sampled_data)
    df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Verification and logging
    subset_stats = df_subset["type"].value_counts()
    print(f"Final subset: {len(df_subset)} examples")
    for q_type, count in subset_stats.items():
        percent = count / len(df_subset) * 100
        full_percent = full_counts.get(q_type, 0) / total_full_samples * 100
        print(
            f"{q_type} ratio: {percent:.1f}% ({count}) | Original: {full_percent:.1f}%"
        )

    # Return as a list of dictionaries
    return df_subset.to_dict("records")


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


def process_nq(dataset: Dataset) -> list:
    filtered_dataset = dataset.filter(
        lambda example: any(
            len(inner_list) > 0 for inner_list in example["short_answers"]
        )
    )

    mini_dataset = filtered_dataset.shuffle(seed=CONFIG.seed).select(
        range(CONFIG.dataset_size)
    )

    return mini_dataset.to_list()


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    dataset_name = CONFIG.dataset_name.lower()

    if "hotpot" in dataset_name:
        dataset = load_dataset(
            CONFIG.dataset_name,
            CONFIG.config,
            split=CONFIG.split,
            trust_remote_code=True,
        )
        subset = process_hotpot(dataset)

    elif (
        "2wikimultihopqa" in dataset_name.lower()
        or "2wikimultihop" in dataset_name.lower()
    ):
        subset = process_2wikimultihop_local(
            rf"..\data\wikimultihop_wiki-processed\{CONFIG.split}.json"
        )

    elif "boolq" in dataset_name:
        dataset = load_dataset(
            CONFIG.dataset_name, split=CONFIG.split, trust_remote_code=True
        )
        subset = process_boolq(dataset)

    elif "natural_questions":
        dataset = load_dataset(
            CONFIG.dataset_name, split=CONFIG.split, trust_remote_code=True
        )
        subset = process_nq(dataset)
    else:
        print(f"Error: No processor found for dataset: {CONFIG.dataset_name}")
        return

    if subset:
        # Construct output file path
        output_name = f"{CONFIG.dataset_name.split('/')[-1]}_mini_{'dev_' if CONFIG.split == 'dev' else ''}{N_SAMPLES}.jsonl"
        output_path = CONFIG.data_dir / output_name

        save_jsonl(subset, output_path)
        print(f"\nSubset successfully created and saved to: {output_path}")


if __name__ == "__main__":
    main()
