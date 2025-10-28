import pandas as pd
import json
import os
import math

# --- Configuration ---
# You MUST update this path to where your downloaded 'dev.json' file is located.
FILE_PATH = r"data\wikimultihop_wiki-processed\test.json"
OUTPUT_FILE_NAME = r"data\wikimultihop_wiki-processed\test_subset_{size}.json"

# --- Define the Target Subset Size ---
SUBSET_SIZE = 1000  # <--- Change this value for a different subset size


# --- 1. Dynamic Calculation of Target Counts Function ---
def calculate_stratified_counts(full_counts, subset_size, total_full_samples):
    """Calculates the proportional counts for the subset, ensuring the total is exact."""
    # Calculate the float target size for each type
    float_targets = {}
    for q_type, count in full_counts.items():
        ratio = count / total_full_samples
        float_targets[q_type] = ratio * subset_size

    # Round down all counts (initial floor)
    target_counts_floored = {k: math.floor(v) for k, v in float_targets.items()}
    total_rounded_count = sum(target_counts_floored.values())

    # Determine how many remaining samples are needed to reach the subset_size
    remaining_needed = subset_size - total_rounded_count

    # Calculate the fractional part for each type
    fractional_parts = {k: v - target_counts_floored[k] for k, v in float_targets.items()}

    # Sort types by the fractional part in descending order
    # The types with the largest fractional parts get the remaining samples
    sorted_fractions = sorted(fractional_parts.items(), key=lambda item: item[1], reverse=True)

    final_counts = target_counts_floored

    # Assign the remaining needed samples one by one
    for i in range(remaining_needed):
        q_type_to_add = sorted_fractions[i][0]
        final_counts[q_type_to_add] += 1

    return final_counts


# --- 2. Load the Full dataset and Determine Full Counts ---
try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df_full = pd.DataFrame(data)

    # --- DYNAMICALLY CALCULATE FULL COUNTS ---
    FULL_COUNTS = df_full['type'].value_counts().to_dict()
    TOTAL_FULL_SAMPLES = len(df_full)

    print(f"Successfully loaded {TOTAL_FULL_SAMPLES} examples from the full dataset.")
    print("\nFull Dataset Question Type Distribution:")
    print(pd.Series(FULL_COUNTS).to_string())
    print("-" * 50)

except FileNotFoundError:
    print(f"ERROR: File not found at '{FILE_PATH}'.")
    print("Please ensure the FILE_PATH is correct and the 'dev.json' file is present.")
    exit()

# --- 3. Calculate Target Counts based on Dynamic Ratios ---
if SUBSET_SIZE > TOTAL_FULL_SAMPLES:
    print(
        f"Warning: Requested subset size ({SUBSET_SIZE}) is larger than the full dataset size ({TOTAL_FULL_SAMPLES}). Setting subset size to max.")
    SUBSET_SIZE = TOTAL_FULL_SAMPLES

TARGET_COUNTS = calculate_stratified_counts(FULL_COUNTS, SUBSET_SIZE, TOTAL_FULL_SAMPLES)

print(f"Calculated Target Subset Counts for {SUBSET_SIZE} samples:")
print(pd.Series(TARGET_COUNTS).to_string())
print("-" * 50)

# --- 4. Perform Stratified Sampling ---
sampled_data = []

print("Performing stratified sampling...")

for q_type, count in TARGET_COUNTS.items():
    # Filter the DataFrame for the current question type
    df_type = df_full[df_full['type'] == q_type]

    # Check if enough samples are available (should only be an issue if SUBSET_SIZE > TOTAL_FULL_SAMPLES, handled above)
    if len(df_type) < count:
        print(
            f"Error: Not enough samples available for '{q_type}'. Available: {len(df_type)}, Requested: {count}. This should not happen if logic is correct.")
        sample = df_type
    else:
        # Randomly sample the required number of instances
        sample = df_type.sample(n=count, random_state=42)

    sampled_data.append(sample)
    print(f"-> Sampled {len(sample)} instances of type '{q_type}'.")

# Combine all the sampled DataFrames
df_subset = pd.concat(sampled_data)

# Randomly shuffle the final subset for randomness in the file
df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

# --- 5. Verify and Save the Subset ---
subset_stats = df_subset['type'].value_counts()
subset_list = df_subset.to_dict('records')

# Update the output file name with the actual size
final_output_file = OUTPUT_FILE_NAME.format(size=len(df_subset))

# Save the new subset to a JSON file
with open(final_output_file, 'w', encoding='utf-8') as f:
    json.dump(subset_list, f, indent=4)

print("-" * 50)
print(f"Subset creation complete. Total samples: {len(df_subset)}")
print(f"New subset saved to: {os.path.abspath(final_output_file)}")
print("\nFinal Question Type Distribution in Subset:")
print(subset_stats.to_string())