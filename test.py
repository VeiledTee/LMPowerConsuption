import os
import pandas as pd
import json

# Define paths
results_dir = "/home/penguins/Documents/LMPowerConsumption/results"
jsonl_file_path = (
    "/home/penguins/Documents/LMPowerConsumption/data/hotpot_qa_mini_1000_indexed.jsonl"
)


def update_csv_ids():
    # 1. Load the real IDs from the JSONL file into a list
    # The index in this list will correspond to the integer ID in your CSV
    print("Loading original IDs from JSONL...")
    jsonl_ids = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            jsonl_ids.append(data["id"])

    # 2. Iterate through the results directory
    for filename in os.listdir(results_dir):
        if filename.startswith("hotpot") and filename.endswith(".csv"):
            file_path = os.path.join(results_dir, filename)
            print(f"Processing {filename}...")

            # Load the CSV
            df = pd.read_csv(file_path)

            # 3. Replace the 'id' column
            # We use the integer value in the 'id' column as an index for our jsonl_ids list
            try:
                df["qid"] = df["qid"].apply(lambda x: jsonl_ids[int(x)])

                # Save the updated CSV (overwriting the old one)
                df.to_csv(file_path, index=False)
                print(f"Successfully updated {filename}")
            except IndexError:
                print(
                    f"Error: An ID in {filename} exceeded the number of rows in the JSONL."
                )
            except Exception as e:
                print(f"An error occurred with {filename}: {e}")


if __name__ == "__main__":
    update_csv_ids()
